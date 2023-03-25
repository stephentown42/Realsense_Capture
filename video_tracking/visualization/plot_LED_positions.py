""" Draw scatter plot of blue and red LED positions for all, or a subset of frames 


Options:



TO DO: 
    Check that assumed frame rate is correct for all videos (or deal with it if not)

"""


import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from Methods.video_tracking import loading as vload
from Methods.video_tracking import transform as vtran
from Methods.video_tracking import plotting as vplot



def main():

    # Parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--example", help="Example to run", type=str, default='matlab')
    parser.add_argument("-f", "--ferret", help="Experimental subject", type=int, default=1605)
    parser.add_argument("-b", "--block", help="Experimental session", type=str, default='J4-45')
    args = parser.parse_args()       
    
    # Draw example figures
    if args.example == 'dlc':       # DeepLabCut

        dlc_file = '2018-02-23_Track_09-57-47DLC_resnet50_FrontiersLED_HEmatSep1shuffle1_400000.csv'
        dlc_info = vload.get_info_for_deeplabcut_file(dlc_file)

        df = vload.read_deeplabcut_csv(
                '/home/stephen/Data/Task_Switching/head_tracking/LED_positions/dlc',
                dlc_file,
                None)

        vplot.plot_all_positions(df, markersize=1, alpha=0.5, im_size=(720, 480))
   
    elif args.example == 'matlab':

        # Camera speed
        # Use a constant value as I don't trust the metadata below, but needs further investigation (TO DO)
        # fps_table = 'Methods/video_tracking/metadata/video_metadata_shutterInterim.parquet'
        # fps_table = pq.read_table(fps_table, columns=['fps'], filters=[("fnum","=",args.ferret), ("block", "in", [args.block])])

        # assert len(fps_table) == 1
        # fps = fps_table['fps'].to_numpy()[0]
        fps = 30
       
        # Load behavioral trials (has sync information)
        trials = vload.get_trial_times(args.ferret, args.block)


        # Load spike times
        spike_file = '/home/stephen/Data/Task_Switching/spike_times_220606_1808.hdf5'
        
        h5_files = vload.list_recording_files_for_block(args.ferret, args.block)
        spike_times = vload.load_spike_times_for_files(spike_file, h5_files)
        spike_times = vtran.map_spiketimes_2_frametimes(spike_times, trials) 
        spike_counts = vtran.bin_spikes_for_video(spike_times, fps)

        # Load LED positions for each frame
        file_path = '/home/stephen/Data/Task_Switching/head_tracking/LED_positions/LED_data_221005_1454.parquet'
        df = vload.load_matlab_parquet(file_path, args.ferret, args.block)

        df = vtran.filter_for_low_likelihoods(df, threshold=0.2)
        df = vtran.remove_large_separations(df, max_separation=50)
        df = vtran.interpolate_missing_frames(df, nframes=20)
        df = vtran.compute_head_pose(df, method='red')
        df = vtran.add_smoothing(df, width=5)
        df = vtran.compute_speed(df, window=10, fps=fps)

        # # Load timing data
        df['time'] = df['frame'] / fps

        trials = vload.get_trial_times(args.ferret, args.block)
        trial_times = trials['starttime_video'].to_numpy()
        print(trial_times)


        # traj = vtran.get_trial_trajectories(df, fps=fps, 
        #     ev_times=trial_times, window=(-3.0, 3.0))

        # vplot.plot_session_trajectory(df, fps=fps, center=(300,200), trial_times=trials['starttime_video'].to_numpy())
        # vplot.plot_all_positions(df, markersize=1, alpha=0.5, im_size=(640, 480))

        vplot.play_video(
            video_file = f"/home/stephen/Data/Task_Switching/videos/F1605_Snorlax_Block_{args.block}_Vid0.avi", 
            LEDs = df[['frame','head_x','head_y']], 
            start_time = 600, 
            duration = 100,
            trail = 1,
            trials = trials,
            norm_spike_rates = spike_counts * 20,
            save = True)



        # vplot.scatter_at_time(df, fps, trial_times, im_size=(640, 480))


        # plot_trial_trajectory(df, trials, im_size=(640, 480))




if __name__ == '__main__':
    main()