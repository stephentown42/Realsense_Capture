""" 

Refine estimates of visual stimulus onsets estimated using edge detection



"""

import json
from typing import Optional
import logging
from pathlib import Path
import os, sys

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path.cwd()))
from lib import utils


# Get paths from environment
load_dotenv()
local_home = Path(os.getenv('local_home'))

plt.rcParams.update({'font.size': 8})




def fit_linear_model(df):
    """  
    prediction_error_limit: 
    
    Error is calculated as squared-difference between predicted and observed edge time (in seconds) 
    """
    
    # Remove nans and check we have enough data
    mdl_data = df[['starttime', 'edge_time']].dropna()

    if mdl_data.shape[0] < 5:
        raise ValueError('Not enough data')

    # Build linear model linking trial start time to edge time (predicted to decrease with start time)
    lin_mdl = LinearRegression()
    lin_mdl.fit(
        mdl_data['starttime'].to_numpy().reshape(-1, 1), 
        mdl_data['edge_time'].to_numpy().reshape(-1, 1)
        )
    
    # Measure difference between model and observations
    mdl_data['predict_time'] = lin_mdl.predict(mdl_data['starttime'].to_numpy().reshape(-1, 1))
    mdl_data = mdl_data.assign(predict_error = lambda x: (x.edge_time - x.predict_time)**2)

    return mdl_data


def flag_outliers(mdl_data:pd.DataFrame, error_limit:float=0.15):

    mdl_data['outlier'] = mdl_data['predict_error'] > error_limit
    return mdl_data


def mask_outliers(df:pd.DataFrame, mdl_data:pd.DataFrame) -> pd.DataFrame:

    # Mask edge values on trials where prediction error exceeds threshold
    idx = mdl_data[mdl_data['outlier']].index

    df.loc[idx,'edge_idx'] = np.nan
    df.loc[idx,'edge_time'] = np.nan

    # Include copy of predictions (in case we want to use them)
    df['edge_time_pred'] = np.nan
    df.loc[idx,'edge_time_pred'] = mdl_data.loc[idx,'predict_time']

    return df


def plot_model(mdl_data, save_path:Optional[Path]=None) -> None:

    fig, ax = plt.subplots(1,1)

    mdl_data.plot(y='starttime', x='edge_time', ax=ax, label='Observed')
    mdl_data.plot(y='starttime', x='predict_time',c='r', ax=ax, label='Prediction')
    
    mdl_data[mdl_data['outlier']].plot.scatter(y='starttime',x='edge_time', ax=ax, c='orange', label='Outliers')

    ax.invert_yaxis()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():

    # Directories
    file_path = local_home / 'Task_Switching/head_tracking/edge_detect_test'

    # Load file list (only specified files will be refined)
    refinement_file = Path.cwd() / 'Methods/video_tracking/synchronization/refinement_required.json'
    with open(refinement_file, 'r') as rf:
        refine_dict = json.load(rf)['regression']

    # Set up logging
    # logger = logging.getLogger(__name__)
    
    # f_handler = logging.FileHandler(
    #     file_path / datetime.now().strftime('%Y-%m-%d_OnsetRefined_%H-%M-%S.log')
    # )

    # f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')    
    # f_handler.setFormatter(f_format)

    # logger.setLevel(logging.DEBUG)
    # logger.addHandler(f_handler)

    # For each session
    for ferret, block_list in refine_dict.items():
        for block in block_list:
        
            print(f"{ferret} Block_{block}")

            # Check if outlier file already processed
            output_file = f"{ferret}_Block_{block}_outlier_LinReg.csv"
            output_path = file_path / output_file
            
            if output_path.exists():
                continue

            # Load existing results
            input_file = f"{ferret}_Block_{block}.csv"
            df = pd.read_csv(file_path / input_file)

            # Transform
            mdl_data = fit_linear_model(df)
            mdl_data = flag_outliers(mdl_data, error_limit=0.125)

            # Output dataframe without outliers            
            df = mask_outliers(df, mdl_data)
            df.to_csv( output_path, index=False)

            # Include figure showing outliers
            output_image = f"{ferret}_Block_{block}_outlier_LinReg.png"
            plot_model(mdl_data, save_path= file_path/output_image)


            # logger.info(f"F{ferret} {block}: Completed successfully")


if __name__ == '__main__':
    main()