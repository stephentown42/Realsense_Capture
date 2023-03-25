""" We have time-varying signals in which we expect to find a one or a sequence of LED pulses that indicate the onset of a trial

Notes:
    - Every session is independent
    - The number of events in a sequence is between 1 and 5
    - In most cases, the number of events is constant within a session (there are some rare exceptions)
    - The duration of events within a session is constant
    - The duration of events across sessions may not be constant
    - All trials are either visual or audiovisual
    - There are some sessions for which videos ended early (not sure why, but possible technical failures / errors). Values taken from video frames in such cases are replaced with nans.

Information available in the project database for each session
    - stimulus duration
    - number of events
    - trial modality (which can be used to ignore data on trials without visual stimuli)
    - visual stimulus location 
        - ignore data when LED is outside image
        - signals are going to differ between bounding boxes, this may be useful information

# NEXT TO DO
    The Canny edge detector does a good job most of the time, but even then there are some errors.
    Need to introduce checks on edges for cases where a pre-stimulus event catches the onset
"""

import logging
from pathlib import Path
import os, sys
import time

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.mixture import BayesianGaussianMixture

sys.path.insert(0, str(Path.cwd()))
from lib import utils
from lib.canny_edge_detect import cannyEdgeDetector
from Methods.video_tracking import plotting as vplot

# Get paths from environment
load_dotenv()
local_home = Path(os.getenv('local_home'))

plt.rcParams.update({'font.size': 8})


def get_timing_info(fnum:int, block:str) -> dict:
    """ Get the duration of stimuli and the interval between stimuli """

    query = """ 
        SELECT 
            stim_duration,
            stim_interval
        FROM 
            task_switch.ferrets f
        WHERE 
            f.id = %(fnum)s;
    """

    info = utils.query_postgres(query, params={'fnum':fnum, 'block':block})
    return info.to_dict('records')[0]


def get_session_dt(fnum:int, block:str):

    query = """ 
        SELECT * 
        FROM task_switch.sessions s
            WHERE s.ferret = %(fnum)s AND s.block = %(block)s;
    """

    return utils.query_postgres(query, params={'fnum':fnum, 'block':block})


def get_trial_info(fnum:int, block:str):
    """ Get details of the stimulus modality and location on each trial """

    query = """ 
    WITH CTE AS (
        SELECT 
            id as trial_num,
            starttime,
            stim_id
        FROM task_switch.trials t
        WHERE t.session_dt = 
            (SELECT datetime 
            FROM task_switch.sessions s
            WHERE s.ferret = %(fnum)s AND s.block = %(block)s)
    )

    SELECT 
        starttime,
        nstimreps,
        modality,
        visual_location 
    FROM CTE
    INNER JOIN task_switch.stimuli
        ON CTE.stim_id = stimuli.id
    ORDER BY starttime;
    """

    df = utils.query_postgres(query, params={'fnum':fnum, 'block':block})

    # Remove auditory trials
    df = df[df['modality'] != 'Auditory']

    return df.reset_index(names='trial_num')


def detect_leading_edge(img):


    for trial in img:
        idx = np.argmax(img > 1)


class OutlierDetector():
    """ A way of removing edges that are far beyond the predictions of a statistical model """

    def __init__(self, data, min_datapoints:int=3):

        self.data = data

        # Check if enough edges to build a model
        not_nan_idx = data['edge_idx'].dropna().index
        self.enough_data = len(not_nan_idx) >= min_datapoints

    def skip_prediction(self):
        # Use data rather than try to make prediction
        self.data['edge_prediction'] = self.data['edge_idx']

    def fit_linear_model(self):
        # Create predictive model
        not_nan_idx = self.data['edge_idx'].dropna().index

        self.model = LinearRegression().fit(
            self.data.loc[not_nan_idx,'trial_num'].to_numpy().reshape(-1, 1), 
            self.data.loc[not_nan_idx,'edge_idx'].to_numpy().reshape(-1, 1)
            )
    
    def predict_edges(self):
        self.data['edge_prediction'] = self.model.predict(self.data['trial_num'].to_numpy().reshape(-1, 1))
    
    def get_prediction_error(self):
        self.data = (
            self.data.assign(
                prediction_error = lambda x: np.abs(x.edge_idx - x.edge_prediction)
            )
        )
    
    def mask_outliers(self, prediction_error_limit):
        #(values with prediction errors above limit)
        outlier_idx = self.data[self.data['prediction_error'] > prediction_error_limit].index
        self.data.loc[outlier_idx,'edge_idx'] = np.nan



# PLOTTING METHODS (move to own class?)

def set_up_plot(locations, figsize) -> np.array:

    # Create figure with columns for each LED location and extra column for all locations combined
    fig, axs_np = plt.subplots(2, len(locations)+1, **{'figsize':figsize})
    
    # Remove spines
    for ax in axs_np.ravel():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Assign axes columns by location
    axs = {k:axs_np[:,i+1] for i, k in enumerate(locations)}
    
    axs['All'] = axs_np[:,0]
    
    axs_np[1,0].set_xlabel('Time')
    axs_np[1,0].set_ylabel('Trial')

    # make it so images fill axes (is this redundant?)
    [ax[0].set_adjustable("datalim") for ax in axs.values()]       

    return axs



def plot_pixelval_distributions(ax, pixel_vals:np.array, jittered_pixels:np.array) -> None:
    """ Plot histograms of pixel values to show input data for mixed modelling """

    n_original, original_edges = np.histogram(pixel_vals, 100)
    n_jittered, jittered_edges = np.histogram(jittered_pixels, 100)
    
    ax.plot(original_edges[:-1], n_original)
    ax.plot(jittered_edges[:-1], n_jittered)


def plot_estimated_means(ax, mdl_means, mdl_weights) -> None:
    """ Show the locations of means estimated from the mixed model as markers above histograms (drawn by plot_pixel_distributions) """
    
    y_temp = np.full_like(mdl_means, max(ax.get_ylim()))
    ax.scatter(mdl_means, y_temp, alpha=mdl_weights, c='orange')
    
    # Set xticks to locations of means
    if len(mdl_weights) > 1:
        xticks = np.sort( np.squeeze( mdl_means))
    else:
        xticks = mdl_means[0]

    xticklabels = [f"{i:.1f}" for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation = 45, fontsize=6)


def plot_image_data(ax, im_data, tvec, title_str) -> None:
    """ Show image data as heatmap that scales to axis size """

    ax.imshow(im_data, interpolation='nearest', aspect='auto')
    ax.set_title(title_str, fontsize=8)
    ax.set_xlim((0, im_data.shape[1]))

    # Label x axis with times rather than indices
    xtimes = [min(tvec), min(np.abs(tvec)), max(tvec)]
    xticks = [np.argmin(tvec), np.argmin(np.abs(tvec)), np.argmax(tvec)]

    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.1f}" for t in xtimes])


def show_edges(ax, df:pd.DataFrame) -> None:
    """ Plot lines to show the estimated index and, where relevant, the regression model used to remove outliers """

    x1 = df['edge_idx'].to_numpy()
    # x2 = df['edge_prediction'].to_numpy()

    y = np.arange(df.shape[0])

    ax.plot( x1, y, color = 'w', lw=2)
    # ax.plot( x2, y, color = 'r', lw=1)


def process_file(frame_val_file, save_dir, logger:logging.Logger, 
    w_threshold:float=0.1, prediction_error_limit:int=5, remove_outliers:bool=True):
    """  
    frame_val_file:
    save_dir: directory for saving data
    w_threshold: threshold for weighting peaks in distribution of pixel values
    prediction_error_limit: absolute number of frames within which an estimated stimulus onset can differ from a linear regression model (larger values are marked as outliers)
    """

    # Get metadata from file name
    file_info = frame_val_file.stem.split('_')

    fnum = int(frame_val_file.parent.name[1:5])
    block = file_info[2]

    save_name = f"F{fnum}_Block_{block}.png"
    save_path = save_dir / save_name

    if save_path.exists():      # Skip if we've already done this file
        logger.info(f"F{fnum} {block}: already exists and will be skipped")
        return

    # Get trial information
    trial_info = get_trial_info(fnum, block)
    stim_info = get_timing_info(fnum, block)

    if len(trial_info) == 0:
        logger.error(f"F{fnum} {block}: No trial information found in database for this unit")
        return

    # Load data stripped from video
    frame_vals = pd.read_csv(frame_val_file)
    tvec = np.array([float(t[:-1]) for t in frame_vals.columns]) # in seconds

    # Check for indexing mismatch - not sure why this is happening
    if trial_info.index.max() > frame_vals.index.max():

        session_info = get_session_dt(fnum, block)
        logger.warning(f"F{fnum} {block}: Index mismatch detected")
        return

    # Create plot object 
    locations = trial_info['visual_location'].unique()

    axs = set_up_plot(locations, figsize=(12,5))

    # Plot all data
    plot_image_data(axs['All'][0], frame_vals.to_numpy(), tvec, 'All (start)')
    
    # Data pre-processing to equate signals sampled in different bounding boxes
    data_for_canny = []

    for location, loc_data in trial_info.groupby('visual_location'):
    
        # Check for nans and fill where necessary (this can occur when videos end early)
        im_data = frame_vals.iloc[loc_data.index]

        # Apply normalization to image data
        im_data = im_data.to_numpy()
        pixel_vals = im_data.reshape(1,-1)
        pixel_vals = pixel_vals[~np.isnan(pixel_vals), np.newaxis]

        # Apply gaussian blur to data to avoid getting too many peaks in the distribution
        jittered_pixels = pixel_vals + (0.5 * np.random.standard_normal(size=pixel_vals.shape))

        # Plot histogram for later inspection
        plot_pixelval_distributions(axs[location][1], pixel_vals, jittered_pixels)

        if any(pixel_vals > 0.0):       # Skip blank datasets
            
            # Estimate the number of light levels in the data
            mdl = BayesianGaussianMixture(
                n_components=10,                # Add more components than needed and then use weights to filter
                random_state=3418, 
                max_iter=1000
            ).fit(jittered_pixels)
            
            # Threshold means from mixture model to remove means with low weight
            mdl_means = mdl.means_[mdl.weights_ > w_threshold]
            mdl_weights = mdl.weights_[mdl.weights_ > w_threshold]

            if len(mdl_weights) == 0:
                logger.error(f"{fnum} {block}: No means in distribution with weights above threshold {w_threshold}")
                continue
            
            # Plot estimated means
            plot_estimated_means(axs[location][1], mdl_means, mdl_weights)

            # Set NaN values to zero (later steps will mean zero is equivalent to background noise)
            im_data[np.isnan(im_data)] = 0

            # Set values below the lowest mean to zero (i.e. get rid of what we expect to be background )
            im_data -= min(mdl.means_[mdl.weights_ > w_threshold])
            im_data[im_data < 0] = 0

            # Clip the top end of values
            max_pixel_mean = max(mdl.means_[mdl.weights_ > w_threshold])
            im_data[im_data > max_pixel_mean] = max_pixel_mean

            # Map everything in between to a standard range
            im_data = im_data / max_pixel_mean

            # 
            data_for_canny.append(
                pd.DataFrame(im_data, columns=frame_vals.columns, index=loc_data.index)
            )

        # Show image data seperately for this location
        plot_image_data(axs[location][0], im_data, tvec, location)
    
    # TEMP - SKIP IF NO DATA COULD BE FOUND (LIKELY DUE TO INDEXING PROBLEM THAT RESULTS IN WARNING ABOVE)
    if len(data_for_canny) == 0:
        logger.warning(f"F{fnum} {block}: No signal found")
        return

    # Perform edge detection across all locations with sensor data
    data_for_canny = pd.concat(data_for_canny).sort_index()

    canny = cannyEdgeDetector(
        imgs=[data_for_canny.to_numpy()],
        padding=3, 
        sigma=1,
        weak_pixel=100, 
        strong_pixel=255, 
        lowthreshold=0.4, 
        highthreshold=0.7
    )

    img_edges = canny.detect()
        
    # Take the first edge, moving across the image horizontally (i.e. with time around trial onset)
    edge_idx = np.argmax(img_edges[0], axis=1).astype(np.float64)
    edge_time = tvec[edge_idx.astype(int)]

    edge_time[edge_idx == 0.0] = np.nan
    edge_idx[edge_idx == 0.0] = np.nan
    
    # Add to table
    trial_info['edge_idx'] = np.nan         # preassign with nans, in case of trials at locations without video data
    trial_info.loc[data_for_canny.index,'edge_idx'] = edge_idx 
    trial_info.loc[data_for_canny.index,'edge_time'] = edge_time 
    
    # # Identify outliers
    # if remove_outliers:

    #     outlier_detector = OutlierDetector(data=trial_info)

    #     if outlier_detector.enough_data:
    #         outlier_detector.fit_linear_model()
    #         outlier_detector.predict_edges()

    #     else:
    #         logger.warning(f"{fnum} {block}: Not enough data to estimate outliers via regression")
    #         outlier_detector.skip_prediction()

    #     outlier_detector.get_prediction_error()
    #     outlier_detector.mask_outliers(prediction_error_limit)
        
    #     trial_info = outlier_detector.data
    # else:
    #     trial_info['edge_prediction'] = trial_info['edge_idx']      # Hack to avoid changing plotting code below (not very SOLID)

    # # Interpolate edge indices for missing spouts
    # interp_order = 3

    # if sum(trial_info.edge_idx.isna() == False) > interp_order:
    #     trial_info['edge_idx'] = trial_info['edge_idx'].interpolate(method='spline', order=interp_order)
    # else:
    #     trial_info['edge_idx'] = trial_info['edge_idx'].interpolate()   # fall back to linear

    # trial_info['edge_idx'] = trial_info['edge_idx'].fillna(method='backfill')

    # Plot estimated onsets
    plot_image_data(axs['All'][1], data_for_canny.to_numpy(), tvec, 'All (end)')
    show_edges(axs['All'][0], trial_info)
    show_edges(axs['All'][1], trial_info)

    for location, loc_data in trial_info.groupby('visual_location'):
        show_edges(axs[location][0], loc_data) 

    # plt.show()
    plt.savefig(save_path)
    plt.close()

    # Reference time and frame to trigger point
    zero_frame = np.argmin(np.abs(tvec))
    zero_time = min(np.abs(tvec))

    trial_info['edge_idx'] -= zero_frame
    trial_info['edge_time'] -= zero_time 

    # Write results as csv file (to allow for different outlier / interpolation approaches)
    csv_output = save_dir / save_name.replace('.png','.csv')
    trial_info.to_csv(csv_output, index=False)

    logger.info(f"F{fnum} {block}: Completed successfully")



def main():

    # Directories
    frame_val_dir = local_home / 'Task_Switching/head_tracking/StimTriggeredFrameVals/'
    save_dir = local_home / 'Task_Switching/head_tracking/edge_detect_test'

    # Set up logging
    logger = utils.make_logger(save_dir, 'OnsetFromLED', add_datetime=True)

    # Process each CSV file
    for frame_val_file in frame_val_dir.rglob('*Block_J*.csv'):
        
        print(frame_val_file.name)
        # time.sleep(0.1)
        process_file(
            frame_val_file, 
            save_dir,
            logger = logger,
            w_threshold = 0.1,
            remove_outliers = False)
        

if __name__ == '__main__':
    main()