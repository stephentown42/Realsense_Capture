""" The optical character estimates from Tesseract are not perfect.

This script extracts the numeric data and removes those values that are clearly errors"""

from pathlib import Path
import os, re, sys

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

sys.path.insert(0, str(Path.cwd()))
from lib import utils


def list_files_to_process():
    """ Videos within a specific date range were problematic - get information about them """

    query = """ 
    SELECT 
        REPLACE( filename, '.avi','.txt') as src_file,
        REPLACE( REPLACE(filename, '.avi', '.dat'), 'Track', 'FrameTDTsamps') as dst_file
    FROM task_switch.video_files
    WHERE session_dt > '2018-01-31 00:00:00.00'
        AND session_dt < '2018-02-20 00:00:00.00';
    """

    return utils.query_postgres(query)



def main():

    data_dir = Path.cwd() / 'data/text/tdt_timestamps_4_videos'
    files = list_files_to_process()

    # Regex 
    number_pattern = re.compile('\d+')
    nonnum_pattern = re.compile('\D+')

    # For each text file (associated with one session)
    for idx, file_info in files.iterrows():

        # Extend paths
        input_file = data_dir / file_info['src_file']
        output_file = data_dir / file_info['dst_file']
        
        # Read messy strings from text file
        with open(input_file) as f:
            strings = f.readlines()

        # Extract numeric data from strings
        samples = np.empty(len(strings))
        samples[:] = np.nan

        for idx, frame_string in enumerate(strings):
            
            frame_string = frame_string.replace('\n','')
            number_match = number_pattern.match(frame_string)
            nonnum_match = nonnum_pattern.match(frame_string)

            if number_match and not nonnum_match:
                samples[idx] = int(number_match.group(0))


        # Create vector of frame counts and select only those frame counts with observed samples
        xy = np.concatenate([
            np.arange(0, len(samples)).reshape((-1, 1)), 
            samples.reshape((-1, 1))
            ], axis=1)

        xy = xy[~np.isnan(samples),:]
        x = sm.add_constant(xy[:,0])
        y = xy[:,1]

        # Find offset, given a known slope
        b_est = y - (4887.5*x[:,1])
        b_est = b_est[:int(len(y) * 0.8)]       # Chop off last 20% (at some point the tdt stops but the camera keeps rolling, leading to errors at very end)
        b_est = np.median(b_est)

        # Compute residuals from our jerry-rigged model
        y_pred = (4887.5*x[:,1]) + b_est
        y_error = np.log(np.abs(y - y_pred)+1e-12)

        # Filter (remember threshold is log_spaced)
        threshold = 10
        y_filtered = y[y_error < threshold]
        x_filtered = x[y_error < threshold, 1]
        y_error_flt = y_error[y_error < threshold]

        # Create new array with just the good data
        new_samples = np.empty(len(samples))
        new_samples[:] = np.nan
        new_samples[x_filtered.astype(int)] = y_filtered


        # Plot to check results look good    
        fig, ax = plt.subplots()
        ax.scatter(np.arange(0, len(new_samples)), new_samples, s=1, alpha=0.25)

        # Interpolate samples 
        interp_samples = np.interp(
            x = np.arange(0, len(strings)),
            xp = x_filtered,
            fp = y_filtered
        )

        ax.scatter( np.arange(0, len(strings)), interp_samples, s=1, alpha=0.25)

        # ax.plot([0, max(x[:,1])], [b_est, b_est+(max(x[:,1])*4887.5)], '--k')
        # ax.set_ylim([0, b_est+(max(x[:,1])*4887.5)])
        # plt.colorbar(s)
        # plt.show()

        # Replace missing numbers with interpolated values (but keep original ones)
        new_samples[np.isnan(new_samples)] = interp_samples[np.isnan(new_samples)]

        # Pad with zero to ensure file consistency
        new_samples = np.concatenate((np.zeros(1), new_samples))
        ax.scatter( np.arange(0, len(new_samples)), new_samples, s=1, alpha=0.25)

        # Write output data
        plt.savefig( output_file.with_suffix('.png'))
        plt.close()

        np.savetxt( output_file, new_samples, fmt='%.0f')



if __name__ == '__main__':
    main()
