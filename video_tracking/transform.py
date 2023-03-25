""" 
Data tranformation module for manipulating results of LED tracking


Version History:
    2022-10-01: Created by Stephen Town
"""


import argparse
from pathlib import Path
import re, sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib import utils

##################################################################################
# Data Transformation

def estimate_mattrack_confidence(df:pd.DataFrame) -> pd.DataFrame:
    """ Convert model coefficients that provide a goodness-of-fit measure
    # (For reference, 'a' and 'c' refer to coefficients of the guassian model f(x) =  a*exp(-((x-b)/c)^2))
    """

    return (
        df
        .assign(blue_LEDlikelihood = lambda x: (x.blue_xa / x.blue_xc) + (x.blue_ya / x.blue_yc))       # This isn't a likelihood in the statistical sense
        .assign(red_LEDlikelihood = lambda x: (x.red_xa / x.red_xc) + (x.red_ya / x.red_yc))
        .drop(columns=['blue_xa','blue_xc','blue_ya','blue_yc','red_xa','red_xc','red_ya','red_yc','fnum','block'])
        .rename({
            'blue_x': 'blue_LEDx',              # Make column names consistent with deep_lab_cut
            'blue_y': 'blue_LEDy',
            'red_x': 'red_LEDx',
            'red_y': 'red_LEDy',
        }, axis=1)
    )


def filter_for_low_likelihoods(df:pd.DataFrame, threshold:float) -> pd.DataFrame:
    """ Remove tracking values for which estimation was uncertain
    
    Note that estimation methods for Matlab and DeepLabCut are different and 
    thus require different values to assess.
    
     """

    # Get indices of trials to mask, independently for red and blue LEDs
    blue_idx = df['blue_LEDlikelihood'] < threshold
    red_idx = df['red_LEDlikelihood'] < threshold

    for var in ['_LEDx','_LEDy']:
        
        df['blue'+var].mask(blue_idx, inplace=True)
        df['red'+var].mask(red_idx, inplace=True)

    return df


def interpolate_missing_frames(df:pd.DataFrame, nframes:int) -> pd.DataFrame:
    """ 
    
    Estimate the position of LEDs using linear interpolation, with a limit on how
    long missing data is tolerated
     """

    # Check that dataframe includes all frames
    assert df.shape[0] == (1+df['frame'].max() - df['frame'].min())
    
    # Extend dataframe to include all frames 
    # (loaded data only contains data for frames where an estimate was made)
    # df = pd.merge(df,
    #     pd.DataFrame(list(range(df['frame'].max())), columns=['frame']),
    #     how = 'right'
    # )
    # 
    # # Get length of NaN streaks 
    # # (so we don't interpolate over huge lengths of missing data)
    # df['missing'] = df['red_peak'].isna()
    # df['streak_start'] = df['missing'].ne(df['missing'].shift(fill_value=True))
    # df['streak_id'] = df['streak_start'].cumsum()

    # streak_count = (
    #     df[['missing','streak_id']]
    #     .groupby('streak_id')
    #     .sum()
    #     .sort_values(by='missing', ascending=False)
    # )

    # Interpolate LED positions for missing frames
    for var in ['blue_LEDx','blue_LEDy','red_LEDx','red_LEDy']:
        df[var].interpolate(method='linear', limit=nframes, inplace=True)

    # Fill missing likelihood values with zeros
    for var in ['blue_LEDlikelihood','red_LEDlikelihood']:
        df[var].fillna(value=0, inplace=True)

    return df


def remove_large_separations(df:pd.DataFrame, max_separation:int=50) -> pd.DataFrame:
    """ Exclude frames on which the separation between red and blue LEDs exceeds
    a sensible value 
    
    Args:
        df: dataframe containing LED positions for each frame
        max_separation: maximum distance (in pixels) permitted between LEDs

    Returns:
        df: dataframe with nans replaceing LED values in frames with large led separation

    Notes:
        See plot_LED_separation.py and ../images/first_pass_red_blue_distance.png
        for distance distribution across the project and the rationale for setting
        50 as the default max distance """

    # Comput
    df = (
        df.assign(distance = lambda x: np.sqrt((x.blue_LEDy - x.red_LEDy)**2 + (x.blue_LEDx - x.red_LEDx)**2))
    )

    # Replace with nans.
    idx = df[df['distance'] > max_separation].index
    df.loc[idx, ['blue_LEDy','blue_LEDx','red_LEDx','red_LEDy']] = np.nan

    # Return without distance column
    return df.drop(columns=['distance'])


def compute_head_pose(df, method:str='unweighted') -> pd.DataFrame:
    """ Estimate the head-center based on the position of left and right LEDs
    
    Notes:
        The arrangement of LEDs on the head makes it hard to compute head direction 
        without extensive checks, so we've left it out - but if implemented, here 
        would be the best place for it to go.

        Method:
            unweighted: take the middle position between red and blue LEDs
            weighted: take weighted mean, using relative fit strengths (experimental)
     """

    if method == 'unweighted':
        return (
            df
            .assign(headx = lambda x: (x.blue_LEDx + x.red_LEDx)/2)
            .assign(heady = lambda x: (x.blue_LEDy + x.red_LEDy)/2)
            #.assign(head_theta = lambda x: NOT IMPLEMENTED
        )

    elif method == 'weighted':
        return (
            df
            .assign(blue_weight = lambda x: x.blue_LEDlikelihood / (x.blue_LEDlikelihood + x.red_LEDlikelihood))
            .assign(red_weight = lambda x: x.red_LEDlikelihood / (x.blue_LEDlikelihood + x.red_LEDlikelihood))
            .assign(headx = lambda x: (x.blue_LEDx*x.blue_weight + x.red_LEDx*x.red_weight))
            .assign(heady = lambda x: (x.blue_LEDy*x.blue_weight + x.red_LEDy*x.red_weight))
            .drop(columns=['blue_weight','red_weight'])
        )

    elif method == 'red':
        return (
            df
            .assign(head_x = lambda x: x.red_LEDx)
            .assign(head_y = lambda x: x.red_LEDy)
        )

    elif method == 'blue':
        return (
            df
            .assign(head_x = lambda x: x.blue_LEDx)
            .assign(head_y = lambda x: x.blue_LEDy)
        )


def invert_affine_warp(M:np.array) -> np.array:
    """ Invert the warp matrix generated by image registration 
    
    Note that results differ slightly from cv2.invertAffineTransform(), but look better when inspecting alignment of positions. Not currently sure why, though I haven't looked into it in great detail
    """

    # Invert scaling
    M[0,0] = 1 / M[0,0]    
    M[1,1] = 1 / M[1,1]

    # Invert translation
    M[0,2] = -M[0,2]
    M[1,2] = -M[1,2]

    # Invert rotation (not sure if this is right)
    temp = M[0,1]
    M[0,1] = M[1,0]
    M[1,0] = temp

    return M


def custom_transform(pts:np.array, M:np.array, invert:Optional[bool]=True) -> np.array:
    """ 
    Perform affine transformation using matrix generated by image alignment pipeline

    Args:
        pts: vector of x,y coordinates in original videos, shape = (m, 2)
        M: affine transformation generated by image alignment pipeline, shape = (2, 3)
        invert: whether to invert the warp transformation matrix (similar to CV2.WARP_INVERSE_MAP)

    Returns:
        pts: vector of x, y coordinates in aligned reference frame

    Notes:
        The warp matrix is of the style compatible with cv2.warpAffine(), however the current function works on point estimates rather than full images
    """

    # Rotate & Scale
    if invert:
        M[0,0] = 1 / M[0,0]     # Invert scaling
        M[1,1] = 1 / M[1,1]

        # temp = M[0,1]
        # M[0,1] = M[1,0]
        # M[1,0] = temp

    pts = pts @ M[:2,:2]
    
    # Translate
    if invert:
        pts[:,0] -= M[0,2]      
        pts[:,1] -= M[1,2]

    else:
        pts[:,0] += M[0,2]      
        pts[:,1] += M[1,2]


    return pts
    
    

def transform_positions_in_dataframe(df:pd.DataFrame, M:np.array, idx:Optional=None) -> pd.DataFrame:
    """ 
    Perform affine transformation using matrix generated by image alignment pipeline

    Args:
        df: dataframe containing x and y positions for multiple landmarks
        M: affine transformation generated by image alignment pipeline, shape = (2, 3)

    Returns:
        df: dataframe containing aligned positions for each landmark

    Notes:
        The function is written to specifically update values in an existing dataframe, and thus hopefully reduce time
        spent copying sets of data.

        The warp matrix is of the style compatible with cv2.warpAffine(), however the current function works on point estimates rather than full images

        From the OpenCV documentation (https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html):
        "The function warpAffine transforms the source image using the specified matrix:
            dst(x,y)=src(M11x+M12y+M13,M21x+M22y+M23)
        when the flag WARP_INVERSE_MAP is set."
    """

    # Only columns containing values greater than zero are eligble (others are ignored)
    columns = [k for (k, v) in dict(df.loc[idx].all()).items() if v]

    # Identify landmarks for mapping
    landmarks = [c.replace('likelihood','') for c in columns if 'likelihood' in c]

    for lm in landmarks:

        df.loc[idx, lm+'x'] = (df.loc[idx, lm+'x'] * M[0,0]) + (df.loc[idx, lm+'y'] * M[0,1]) + M[0,2]
        df.loc[idx, lm+'y'] = (df.loc[idx, lm+'x'] * M[1,0]) + (df.loc[idx, lm+'y'] * M[1,1]) + M[1,2]




def add_smoothing(df, width=5) -> pd.DataFrame:
    """ Smooth using rolling median """

    return df.rolling(window=width, on='frame').median()


def compute_speed(df, window:int, landmark:str)-> pd.DataFrame:
    """ Get speed of landmark movement 
    
    Args:
        df: dataframe containing time, frame and x and y positions for landmark(s)
        window: number of frames over which speed is computed, must be even
        landmark: name of landmark (e.g. 'head','red_LED' etc.)

    Returns 
        dataframe with additional columns for speed of landmark movement and the average frame for which speed was computed
    """

    if window % 2 != 0:
        raise ValueError("window must be an even number")

    cols = ['frame', landmark+'x', landmark+'y','time']

    sp = pd.merge(
        df[cols],
        df[cols].shift(window),
        left_index=True,
        right_index=True,
        suffixes=('_start','_end')
    )

    sp = (
        sp
        .assign(speed_frame = lambda x: (x.frame_start + x.frame_end)/2)
        .dropna()
        .rename(columns = {landmark+s:s for s in ['x_end','x_start','y_end','y_start']})
        .assign(delta_x = lambda x: x.x_end - x.x_start)
        .assign(delta_y = lambda x: x.y_end - x.y_start)
        .assign(duration = lambda x: x.time_end - x.time_start)
        .assign(displacement = lambda x: np.sqrt(x.delta_x**2 + x.delta_y**2))
        .assign(speed = lambda x: x.displacement / x.duration)
        .rename({'speed':landmark+'speed'}, axis=1)
    )

    # Add speed information back into main dataframe
    return pd.merge(df, sp[['speed_frame',landmark+'speed']], how='left', left_on='frame', right_on='speed_frame')


def compute_acceleration(df, window:int, landmark:str='head')-> pd.DataFrame:
    """ Get acceleration of marker 

    Args:
        df: dataframe containing time, frame and speed of landmark(s)
        window: number of frames over which acceleration is computed, must be even
        landmark: name of landmark (e.g. 'head','red_LED' etc.)

    Returns 
        dataframe with additional columns for acceleration of landmark and the average frame for which acceleration was computed
    """

    if window % 2 != 0:
        raise ValueError("window must be an even number")

    cols = ['frame', landmark+'speed', 'time']

    sp = pd.merge(
        df[cols],
        df[cols].shift(window),
        left_index=True,
        right_index=True,
        suffixes=('_start','_end')
    )

    sp = (
        sp
        .assign(acceleration_frame = lambda x: (x.frame_start + x.frame_end)/2)
        .dropna()
        .rename(columns = {landmark+s:s for s in ['speed_end','speed_start']})
        .assign(duration = lambda x: x.time_end - x.time_start)
        .assign(delta_speed = lambda x: x.speed_end - x.speed_start)
        .assign(acceleration = lambda x: x.delta_speed / x.duration)
        .rename({'acceleration':landmark+'acceleration'}, axis=1)
    )

    # Add speed information back into main dataframe
    return pd.merge(df, sp[['acceleration_frame', landmark+'acceleration']],
        how='left', left_on='frame', right_on='acceleration_frame')



def get_trial_trajectories(LEDs:pd.DataFrame, fps:int, ev_times:np.array, 
    window=Tuple[float, float]) -> np.array:
    """ Cross-reference times of trials with LED position records to get
    the x and y positions in time window around each trial
    
    Args:
        LEDs: LED positions for every frame
        fps: frame rate
        trials: array of event times
        window: time window around which to get LED positions

    Returns:
        traj: tensor with dimensions for trial, time (sample) and spatial dimension (x or y)
    
      """

    # Convert times to samples for quick indexing
    window_samps = [np.round(x*fps) for x in window]
    window_duration = int(window_samps[1] - window_samps[0])
    ev_samps = np.round( ev_times * fps)

    # Preassign results matrices
    traj = np.zeros((len(ev_samps), window_duration, 2), dtype=float)
    
    # For each event (trial)
    for i, t in enumerate(ev_samps):

        idx = LEDs[(LEDs['frame'] >= t+window_samps[0]) & (LEDs['frame'] < t+window_samps[1])].index

        if len(idx) > 0:
            traj[i,:,0] = LEDs['head_x'].loc[idx].to_numpy()
            traj[i,:,1] = LEDs['head_y'].loc[idx].to_numpy()

        else:
            print(f"Could not find data for trial {i}")
    
    return traj


def map_TDTev_2_frametimes(x, df):
    """ Estimate times of tdt events (e.g. sensors, stimuli etc) in videos
    
    Args:
        x: array of event times
        df: trial timestamps according to the frame and tdt systems clocks
    

    TO DO: 
        1. Bring lag into the calculation...
            Thought experiment: we have two sensor events at 10 and 20 seconds, and 5 dropped frames between 15 and 16 seconds
            If we assume a constant fps of 30 seconds, then the two events would be mapped to frames 300 and 600 respectively, which would be wrong
            The correct answer would be to map the events to frame 300 and 595

        2. Express event times as frames (not seconds) - this will make it easier to reference when replotting the video n 

     Our ability to match the times of video frames and tdt/mcs clocks is determined by common markers in the two time-lines (e.g. onset of visual trials, in which the stimulus LEDs are bright in the video). For each trial we have the chance to update our estimate of the correspondance between video and alternative clock. 
     
     In this function, we therefore divide the timecourse of a behavioral session into time bins associated with each trial, for which we also have an estimate of the difference between video and clock time. We then look for input events (which could be spikes, stimuli, sensors etc) within each bin and apply the relevant correction.
    """

    # Time bin edges according to original clock 
    boundaries = [0] + df['starttime'].to_list() + [float('inf')]   # shape = ntrials+2

    # Correction to make for each boundary
    correction = df['starttimecorrected'] - df['starttime'] 
    correction = correction.to_list()
    correction = [correction[0]] + correction                       # shape = ntrials+1

    # For every trial
    for i, delta_t in enumerate(correction):
        x[(x >= boundaries[i]) & (x < boundaries[i+1])] += delta_t

    return x


def map_spiketimes_2_frametimes(spike_times:dict, df:pd.DataFrame) -> np.array:
    """  
    Express spike times relative to video clock

    df: trial timestamps according to the frame and multichannel systems clocks
    """
    
    # Boundaries in which to consider spikes
    mcs = [0] + df['starttime_mcs'].to_list() + [float('inf')]
    
    # Correction to make for each boundary
    correction = df['starttime_video'] - df['starttime_mcs'] 
    correction = correction.to_list()
    correction = [correction[0]] + correction

    # Define update function
    def update_spiketimes(x):
        for i, delta_t in enumerate(correction):
            x[(x >= mcs[i]) & (x < mcs[i+1])] += delta_t

        return x

    # Return updated spike times for each channel
    return {k:update_spiketimes(v) for (k, v) in spike_times.items()}


def bin_spikes_for_video(spike_times:dict, fps) -> dict:
    """ Convert individual spike times into spike counts and the map spike count
    to integer values for plotting in video """

    # Get maximum time
    max_t = max([max(v) for v in spike_times.values()])

    # How many standard deviations we want to map to color values
    # (Values above or below -sd range will be clipped)
    std_dev_range = 3
    
    # Get bin edges
    bin_edges = np.arange(0., np.ceil(max_t), 1/fps)

    # Preassign output array
    n_bins = len(bin_edges)-1
    n_chans = len(spike_times)
    spike_count = np.empty((n_chans, n_bins), dtype=np.int8)

    # For each channel
    for chan, ev_t in spike_times.items():
    
        # Get channel number (e.g. APC01 = 1)   
        chan_idx = int(chan[3:]) - 1            # convert 1-based to zero-based   
        # Could put channel mapping here
        hist, _ = np.histogram(ev_t, bin_edges)

        # Map to pixel values
        hist[hist > 255] = 255
        spike_count[chan_idx,:] = hist.astype(np.uint8)
                
    return spike_count
    


def main():
    pass



if __name__ == '__main__':
    main()