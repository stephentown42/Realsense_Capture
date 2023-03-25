""" 
Apply correction computed from image registration to DLC tracking results

Requires connection to project database

Version History:
    2023-02-15: Created by Stephen Town

"""

from datetime import datetime
import logging
from pathlib import Path
import os, sys

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path.cwd()))

from lib import utils
from Methods.video_tracking import loading as vload
from Methods.video_tracking import transform as vtran


def get_img_warp():
    """ Get transformation matrix for aligning images to a common reference frame """

    query = """ 
    SELECT 
        ferret as fnum,	block, video_file, calib_image, registration_by as src,
        rotation_matrix_11 as r11,
        rotation_matrix_12 as r12,
        rotation_matrix_21 as r21,
        rotation_matrix_22 as r22,
        translation_column as tx,
        translation_row as ty
    FROM task_switch.video_calib_images vc
    INNER JOIN task_switch.calibration_images ci
        ON vc.calib_image = ci.id
    INNER JOIN task_switch.video_files vf
        ON vc.video_file = vf.filename;
    """

    return utils.query_postgres(query)


def main():
    
    # Load tracking data
    load_dotenv()
    data_dir = Path(os.getenv("local_home")) / 'Task_Switching/head_tracking'

    dlc_filepath = data_dir / 'DLC_combined_230215_1644.parquet'
        
    blocks = vload.get_unique_rows_from_parquet( dlc_filepath, columns=['fnum','block'])
    blocks['track'] = True

    # Get image registration for each video from database
    img_reg = get_img_warp()

    # Link tracking results to alignment data
    dlc_reg = pd.merge(
        left = blocks, 
        right = img_reg, 
        left_on = ['fnum', 'block'],
        right_on = ['fnum','block'],
        how='outer')

    # OUTPUTS ##########################################################
    # Progress log
    logger = utils.make_logger(data_dir, 'DLC_align', add_datetime=True)

    # Report any blocks for which there is no calibration data
    track_no_calib = dlc_reg.query('tx.isna()')
    
    if track_no_calib.shape[0] > 0:
        logger.info(f"Discarding {track_no_calib.shape[0]} tracked blocks without calibration info linked")
    
        for i, row in track_no_calib.iterrows():
            logger.info(f"F{row['fnum']}, Block_{row['block']}")

        dlc_reg = dlc_reg.drop(index=track_no_calib.index)
    
    # Report any calibration data for which there is no tracking data
    calib_no_track = dlc_reg.query('track.isna()')

    if calib_no_track.shape[0] > 0:
        logger.info(f"Discarding {calib_no_track.shape[0]} calibration links without dlc data")

        for i, row in calib_no_track.iterrows():
            logger.info(f"F{row['fnum']}, Block_{row['block']}")

        dlc_reg = dlc_reg.drop(index=calib_no_track.index)

    # Set up parquet file for output
    schema = pq.read_schema(dlc_filepath)
    save_path = data_dir / datetime.now().strftime('DLC_aligned_%y%m%d_%H%M.parquet')        

    with pq.ParquetWriter(save_path, schema, compression='gzip') as writer:

        # For each calibration image (which will have a constant warp matrix)
        for (ferret, block), block_df in dlc_reg.groupby(['fnum','block']):       

            logger.info(f"Processing F{ferret}, Block_{block}")

            # Get the correction matrix, check it's is unique and then reshape
            warp_mat = block_df[['r11','r12','tx','r21','r22','ty']].drop_duplicates()
            assert warp_mat.shape[0] == 1

            warp_mat = warp_mat.to_numpy().reshape((2,3))
            
            warp_inv = vtran.invert_affine_warp(warp_mat.copy())

            # Load dlc tracking data 
            dlc_data = vload.load_parquet(dlc_filepath, fnum=ferret, block=block)

            # Update the data held in memory using alignment matrix
            vtran.transform_positions_in_dataframe( dlc_data, warp_inv, dlc_data.index)
                
            # Write as individual blocks 
            writer.write_table(
                pa.Table.from_pandas(dlc_data, preserve_index=False)
            )


if __name__ == '__main__':
    main()