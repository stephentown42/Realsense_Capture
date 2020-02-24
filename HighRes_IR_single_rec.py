'''
HighRes_IR_single_rec.py

Stephen Town: 08 Feb 2020
'''

import pyrealsense2 as rs
import numpy as np
import time
import cv2
import logging

from win32com.client import *


def configure_camera(rs, logging, im_y, im_x, fps):

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.infrared, 1, im_x, im_y, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, im_x, im_y, rs.format.y8, fps)

    profile = pipeline.start(config)

    sensor = profile.get_device().first_depth_sensor()
    sensor.set_option(rs.option.emitter_enabled, 0)  # Disable IR emitter

    s = profile.get_device().query_sensors()[0]
    s.set_option(rs.option.enable_auto_exposure, 1)  # Set exposure

    logging.info('Camera successfully started')

    try:
        s.set_option(rs.option.gain, 100)

    except Exception as e:
        logging.error('Error occurred ' + str(e))

    return pipeline


def connect_to_TDT():

    tdt = Dispatch('TDevAcc.X')
    tdt.ConnectServer('Local')
    tdt.SetTargetVal('RX8.gainError', 0)

    logging.info("TDT connection formed")
    return tdt


def create_video_out(cv2, logging, file_path, im_y, im_x, fps, camera):

    vid_file = file_path + '_AT' + str(camera) + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(vid_file, fourcc, fps, (im_x, im_y))

    logging.info("Videowriter successfully created")
    return out


def get_center_value(img, pts):

    intensities = img[pts[0], pts[1], 1]
    center_val = intensities.mean()

    return center_val


def add_text(cv2, im, pos, prefix, value, color=(255, 255, 255)):

    txt = prefix + ': ' + str(round(value))
    cv2.putText(im, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def create_metadata_text(file_path, var_names):

    fid = open(file_path + '_AT.txt', 'w')

    for i in range(0, len(var_names) - 1):
        fid.write(var_names[i] + '\t')

    fid.write(var_names[-1] + '\n')    # Finish with new line

    return fid


def main():

    # Options
    fps = 30    # frames per second
    im_x = 1280  # image width in pixels
    im_y = 720  # image height in pixels

    file_path = time.strftime('D:/Python_Videos/%Y-%m-%d_Track_%H-%M-%S')

    var_names = ['FrameCount','TimeStamp','FrameTs','SensorTs','ArrivalTs','TDT_Sample']
    fid = create_metadata_text(file_path, var_names)

    logging.basicConfig(filename=file_path + '.log', level=logging.INFO)

    pipeline = configure_camera(rs, logging, im_y, im_x, fps)

    tdt = connect_to_TDT()

    vid_1 = create_video_out(cv2, logging, file_path, im_y, im_x, fps, 1)

    logging.info("Writing frames")

    ###############################
    while(True):

        # Read in image, convert to compatible data format for saving
        frames = pipeline.wait_for_frames()

        tdt_sample = tdt.GetTargetVal('RX8.zTime')
        frame_ts = frames.get_frame_metadata(rs.frame_metadata_value.frame_timestamp)
        sensor_ts = frames.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)
        arrival_ts = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)

        ir_frame = frames.get_infrared_frame(1)
        ir_image = np.asanyarray(ir_frame.get_data())
        ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

        add_text(cv2, ir_image, (10, 630), 'Frame', frames.frame_number)
        add_text(cv2, ir_image, (10, 650), 'Timestamp', frames.timestamp)
        add_text(cv2, ir_image, (10, 670), 'Frame', frame_ts)
        add_text(cv2, ir_image, (10, 690), 'Sensor', sensor_ts)
        add_text(cv2, ir_image, (10, 710), 'Arrival', arrival_ts)
        add_text(cv2, ir_image, (10, 20), 'TDT', tdt_sample)

        # Write the frame to disk
        vid_1.write(ir_image)

        # Write the time to text file
        fid.write('%d\t' % frames.frame_number)
        fid.write('%d\t' % frames.timestamp)
        fid.write('%d\t' % frame_ts)
        fid.write('%d\t' % sensor_ts)
        fid.write('%d\t' % arrival_ts)
        fid.write('%d\n' % tdt_sample)

        # Show the user
        cv2.imshow('RealSense', ir_image)

        # Escape once tdt closes
        if tdt.GetSysMode() == 0:
            break

    # Release everything if job is finished
    fid.close()
    pipeline.stop()
    vid_1.release()
    vid_2.release()
    cv2.destroyAllWindows()
    tdt.CloseConnection()


if __name__ == '__main__':
    main()
