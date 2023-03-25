""" 
Final post-processing step after alignment and synchronization in which landmark positions with low likelihoods are removed and, where possible, replaced with linear interpolation

Interpolation is performed separately for each landmark

 """

from datetime import datetime
import os, sys
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


load_dotenv()
sys.path.insert(0, os.getenv("repo_path"))

from lib import utils
from Methods.video_tracking import loading as vload


def mask_landmark_data(df:pd.DataFrame, landmark:str, threshold:float):
    """ Replace x and y positions with nans when likelihood falls below a specific value """

    index = df[df[landmark+'likelihood'] < threshold].index
    df.loc[index, landmark+'x'] = np.nan
    df.loc[index, landmark+'y'] = np.nan
    
    return df, len(index)


def get_default_fps(fnum:int, block:str):

    # List videos associated with block
    query = """        
        SELECT 
	        rv2, 
            default_fps
        FROM task_switch.video_files
        WHERE 
            ferret = %(fnum)s 
            AND block = %(block)s; """

    df = utils.query_postgres(query, params={'fnum':int(fnum), 'block':block})

    # If RV2 videos exist, focus on them
    if df.rv2.any():
        df = df[df['rv2']==True]

    # Return max frame rate for data
    return df['default_fps'].max()


def main():

    # Settings
    threshold = 0.6

    interp_kw = dict(                       # Args for DataFrame.interp
        method = "linear",
        limit_direction = 'both',
        limit_area = "inside"
    )

    # Load environmental variables and define paths
    data_path = Path(os.getenv("local_home")) / 'Task_Switching/head_tracking'
    dlc_file = 'DLC_alignsync_230302_1511.parquet'

    input_path = data_path / dlc_file
    blocks = vload.get_unique_rows_from_parquet( input_path, ['fnum','block'])

    # List landmarks
    schema = pq.read_schema( input_path)
    landmarks = [x.replace('likelihood','') for x in schema.names if 'likelihood' in x]

    # Progress log
    log_filename = 'DLC_alignSyncInterp'
    logger = utils.make_logger(data_path, log_filename, add_datetime=True)

    logger.info(f"Threshold for removing data = {threshold}")
    logger.info(f"Interpolation settings: {interp_kw}")

    # Open parquet writer for saving multiple results
    save_path = data_path / datetime.now().strftime('DLC_alignSyncInterp_%y%m%d_%H%M.parquet') 
   
    with pq.ParquetWriter(save_path, schema, compression='gzip') as writer: 

        # For each block
        for _, block in blocks.iterrows():

            # Max interpolation length needs to be different for videos with low (10fps) and high (30 fps) frame rates
            if get_default_fps(block['fnum'], block['block']) > 25:
                interp_kw['limit'] = 15
            else:
                interp_kw['limit'] = 5

            logger.info(f"Running F{block['fnum']} Block_{block['block']} - interpolation limit = {interp_kw['limit']}")

            dlc_df = vload.load_parquet(input_path, block['fnum'], block['block'])

            # Mask and interpolate separately for each landmark
            for landmark in landmarks:

                if dlc_df[landmark+'likelihood'].mean() == 0.0:         # All zeros indicates this is placeholder data that we can skip
                    continue

                dlc_df, n_mask = mask_landmark_data(dlc_df, landmark, threshold)
                
                p_mask = (n_mask / dlc_df.shape[0]) * 100
                logger.info(f"{landmark}: {n_mask} of {dlc_df.shape[0]} samples ({p_mask:.2f} %) masked")

                dlc_df[landmark+'x'] = dlc_df[landmark+'x'].interpolate(**interp_kw)
                dlc_df[landmark+'y'] = dlc_df[landmark+'y'].interpolate(**interp_kw)

            # Write to new file
            writer.write_table( pa.Table.from_pandas( dlc_df, preserve_index=False))

if __name__ == '__main__':
    main()