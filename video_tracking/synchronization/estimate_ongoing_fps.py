""" 
Estimate Onoing Frame Rate (FPS)

We don't know exactly when a frame is dropped, but we can at least estimate the time window in which a frame is lost using the times of trial onset, synchronized by monitoring the visual stimulus within the image.

The visual stimulus at trial onset serves as a reference point between clocks and so we know the time between trial onsets, as well as the number of frames. This allows us to estimate the frames per second in periods across the session.

We expect that the estimated fps will not be constant and so we call our estimate the estimated ongoing fps.

Note: The code below was a first guess at how to calculate this for a single session. However I think if we take a step back and look at the output of the matlab code that aligns trials and calculates lags, we should find a more appropriate dataset to perform the underlying estimation (with fewer rounding errors)


Version History:
    2022-Dec-04: Created, Stephen Town
    2022-Dec-28: Updated to use results of canny detection 
"""

from pathlib import Path
import os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from lib import utils

# Load environmental variables
from dotenv import load_dotenv
load_dotenv()
load_dotenv(dotenv_path=Path('.')/'.env')


def get_trials_for_all_sessions() -> pd.DataFrame:
    """ 
    Query database for all trials in the project for which we have neural data

    Args:
        fnum: id of subject
        block: suffix for session identifier

    >>> get_trials_for_all_sessions()
    
    TO DO: Improve query to avoid returning duplicates
     """
 
    query = """
        SELECT 
            s.ferret,
            s.block,
            s.datetime,
            mcs.starttime as starttime_orig,
            mcs.starttimecorrected,
            vt.starttime_video
        FROM task_switch.sessions s
        INNER JOIN task_switch.mcs_trials_20230219 as mcs 
            ON s.datetime = mcs.session_dt
        INNER JOIN task_switch.video_timestamps vt
            ON s.datetime = vt.session_dt AND mcs.starttime = vt.starttime
        ORDER BY
            s.datetime, 
            mcs.starttimecorrected;
        """

    return utils.query_postgres(query)
    


def get_trials_for_session(fnum:int, block:str) -> pd.DataFrame:
    """ 
    Query database for trials associated with a specific session

    Args:
        fnum: id of subject
        block: suffix for session identifier

    >>> get_trials_for_session(1602, 'J2-17')
     """
 
    query = """
        SELECT
            starttime as starttime_orig,
            starttimecorrected
        FROM task_switch.mcs_trials_20230219
        WHERE session_dt = (SELECT
                                datetime
                            FROM task_switch.sessions
                            WHERE ferret = %(fnum)s 
                                AND block = %(block)s);
        """

    return utils.query_postgres(query, params={'fnum':fnum, 'block':block})


def get_ongoing_frame_rate(behavior:pd.DataFrame, lag:pd.DataFrame, default_frame_rate:int) -> pd.DataFrame:
    """ Calculate onoing frame rate
    
    Args:
        df: dataframe containing trial onset times with correction and in original format
        lag: dataframe containing the times at which visual stimuli were detected in a video for each trial
    
    Notes:
        The corrected trial onset times represent a more accurate temporal record than the original format. It would better to actually use these corrected times in 'estimate_trial_onset_from_LEDs' but this would require addition of corrected start times to trial tables (something we didn't originally think was needed for anything other than neural analysis.)

     """

    # Merge info and drop irrelavant columns
    df = (
        pd.merge(
            left = behavior, 
            right = lag, 
            how = 'left', 
            left_on = 'starttime_orig', 
            right_on = 'starttime'
        )
        .drop(columns=['starttime','trial_num'])
        .assign(start_frame = lambda x: x.starttimecorrected * default_frame_rate)    # checked that this is consistent with get_video_data.m (line 35)
    )

    # Fill forwards missing edges (could arise due to audio only stimuli, spouts out of view etc.)
    # Then fill backwards to catch any missing values at start of data (we don't want to do this before ffill)
    df['edge_idx'].fillna(method='ffill', inplace=True)
    df['edge_idx'].fillna(method='bfill', inplace=True)     # 

    # Apply correction to start frame (edge index should be negative as the visual stimulus arrives before it's expected thanks to dropped frames)
    df['start_frame'] = df['start_frame'] + df['edge_idx']

    # Get frame and time of previous trial
    # NOTE: using fill_value=0 assumes that the TDT and video clocks have a common onset
    offset_frame = df['edge_idx'].iloc[0]           # The frame at which the tdt clock was zero (positive if video started before tdt, negative if tdt started before video)
    df['prev_start_frame'] = df['start_frame'].shift(1, fill_value=offset_frame)
    df['prev_start_time'] = df['starttimecorrected'].shift(1, fill_value=0)

    # Calculate trial duration and number of frames, to then get fps
    df = (
        df
        .assign( duration = lambda x: x.starttimecorrected - x.prev_start_time)
        .assign( n_frames = lambda x: x.start_frame - x.prev_start_frame)
        .assign( fps = lambda x: x.n_frames / x.duration)
        .query( "duration > 0.01")           # This happens when the trial timestamps associated with two MCS files are not identical
    )



    return df


def find_onset_estimates(file_path:Path, fnum:int, block:str, priority_list:dict):
    """ Multiple versions of onset estimate results may exist, for example if outliers have been removed. This function returns the file name with highest priority """

    # For each type of edge file
    for file_stub in priority_list.values():

        # Check if path exists
        file_name = f"F{fnum}_Block_{block}{file_stub}"
        test_file_path = file_path / file_name

        # Return first case that exists
        if test_file_path.exists():
            return test_file_path

    # Return None if not found
    return None


def get_default_fps(fnum:int, block:str):
    """ Query the FPS value in the video file properties for this file """

    query = """    
    SELECT 
        filename,
        default_fps,
        rv2
    FROM task_switch.video_files
    WHERE 
        ferret = %(fnum)s
        AND block = %(block)s;
    """

    default_fps = utils.query_postgres(query, params={'fnum':fnum, 'block':block})

    # Select RV2 videos for any block where that's available
    if any(default_fps['rv2']):
        default_fps = default_fps.query("rv2 == True")
    else:
        # Otherwise there should only be one video, though the file may have been resized (in which case we ignore the resized version)
        idx = default_fps[default_fps.filename.str.contains('_resized')].index
        default_fps = default_fps.drop(index = idx)

    # The dataframe should now only contain one video
    assert default_fps.shape[0] == 1
    
    # Return the fps value
    return default_fps['default_fps'].to_list()[0]


def main():

    # Paths
    data_dir = Path(os.getenv("local_home")) / 'Task_Switching/head_tracking'
    edge_val_dir = data_dir / 'edge_detect_test'
    output_dir = data_dir / 'ongoing_fps'

    # Input files come from attempts to estimate trial onsets; a single file may be processed using standard methods, followed in some cases by regression to remove outliers, and manual inspection if the outliers cannot be removed using other means. We therefore create a priority list that gives preference to versions of the results with the most extensive post-processing (manual, then regression) but also allows for the many cases where the standard approach was sufficent.
    # Order of priority (highest to lowest) - python maintains order of insertion
    priority_list = {                                
        'manual_removal': '_manual.csv', 
        'linear_regression': '_outlier_LinReg.csv',             # <- this level and above may exist
        'standard': '.csv'}                                     # <- this should always exist

    # Set up logging
    logger = utils.make_logger(output_dir, 'ongoing_fps', add_datetime=True)
    
    # Load trial info from database
    # fnum = 1613               # Old version for single block
    # block = 'J5-12'
    # all_data = get_trials_for_session(fnum, block)

    # all_data = get_trials_for_all_sessions()
    # all_data = all_data[(all_data['ferret'] == 1506) & (all_data['block'].str.contains('J2-33'))]
    
    # For each edge detection file
    for edge_file in edge_val_dir.glob('*.csv'): 

        logger.info(f"Processing {edge_file.name}")

        # If the file is potentially of a lower status (i.e. not manually checked)
        if '_manual' not in edge_file.stem:
            
            # Skip if a manual file exists
            higher_priority = edge_val_dir / edge_file.name.replace('.csv','_manual.csv')

            if higher_priority.exists():
                logger.info(f"Higher priority (manual file) exists - skipping")
                continue
            else:
                higher_priority = edge_val_dir / edge_file.name.replace('.csv','_outlier_LinReg.csv')

                if higher_priority.exists():
                    logger.info(f"Higher priority (linear regression file) exists - skipping")
                    continue
    
        # Load behavioral data
        name_parts = edge_file.stem.split('_')
        fnum = int(name_parts[0][1:5])
        block = name_parts[2]

        # Load and preprocess behavioral data
        trials = get_trials_for_session(fnum, block).drop_duplicates()

        if trials.shape[0] == 0:
            logger.error(f"Could not find trial times in database for F{fnum} Block_{block}")
            continue

        # Load lag values calculated through edge detection        
        lag = pd.read_csv(edge_file)

        default_fps = get_default_fps(fnum, block)
        df = get_ongoing_frame_rate(trials, lag, default_fps)

        # Write to file
        output_file = f"F{fnum}_Block_{block}.csv"
        df.to_csv(output_dir / output_file, index=False)
        logger.info(f"\n\tSaving {output_file}")

        # Draw scatterplot showing change in lag across session (expected to increase linearly) 
        fig, axs = plt.subplots(2,1, sharex=True)
        
        df.plot.line(x='starttimecorrected', y='fps', ax=axs[0])
        df.plot.scatter(x='starttimecorrected', y='edge_idx', ax=axs[1])
        
        plt.savefig( output_dir / output_file.replace('.csv','.png'))
        # plt.show()
        plt.close()
        logger.info(f"\n\tSaved figure")


if __name__ == '__main__':
    main()