import pyrealsense2 as rs
import numpy as np
import time
import cv2
import logging

from win32com.client import *


def connect_to_TDT():

    tdt = Dispatch('TDevAcc.X')
    tdt.ConnectServer('Local')
    tdt.SetTargetVal('RX8.gainError', 0)

    logging.info("TDT connection formed")
    return tdt


def detect_center_strip(pipeline, cv2, np, logging, threshold=45):

    frames = pipeline.wait_for_frames()

    ir_frame = frames.get_infrared_frame()
    ir_image = np.asanyarray(ir_frame.get_data())

    ret, tImg = cv2.threshold(ir_image, threshold, 255, cv2.THRESH_BINARY_INV)

    im2, contours, hierarchy = cv2.findContours(
        tImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(im2)
    count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 800 and area < 20000:
            cv2.drawContours(mask, contours, count, color=255, thickness=-1)
            logging.info('Contour detected')

        count += 1

    pts = np.where(mask == 255)
    logging.info("Center analysis complete")

    return pts


def get_center_value(img, pts):

    intensities = img[pts[0], pts[1], 1]
    center_val = round(intensities.mean())

    return center_val


def add_text(cv2, im, pos, value, color=(255, 255, 255)):

    txt = str(round(value))
    cv2.putText(im, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():

    # Options
    fps = 30    # frames per second
    im_width = 1280
    im_height = 720

    # Define file name based on current time
    videoName = time.strftime('D:/Python_Videos/%Y-%m-%d_Track_%H-%M-%S.avi')
    textName = videoName.replace('.avi', '.txt')
    logName = textName.replace('.txt', '.log')

    # Open text file to write TDT samples and center values
    fid = open(textName, 'w')
    fid.write('FrameCount\tTimeStamp\tTDT_Sample\tTDT_status\n')

    # Initialize the log settings
    logging.basicConfig(filename=logName, level=logging.INFO)

    # Configure realsense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, im_width, im_height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, im_width, im_height, rs.format.y8, fps)

    # Start streaming
    profile = pipeline.start(config)

    # Disable IR emitter
    sensor = profile.get_device().first_depth_sensor()
    sensor.set_option(rs.option.emitter_enabled, 0)

    # Set exposure
    s = profile.get_device().query_sensors()[0]
    s.set_option(rs.option.enable_auto_exposure, 1)

    logging.info('Camera successfully started')

    pts = detect_center_strip(pipeline, cv2, np, logging)

    tdt = connect_to_TDT()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(videoName, fourcc, fps, (1280, 360))

    # Initialize values for the text being added to frames
    fontColor = (255, 255, 255)

    logging.info("Videowriter successfully created")

    ###########################
    # Set gain - this is where things can go wrong
    try:
        s.set_option(rs.option.gain, 100)

    except Exception as e:
        logging.error('Error occurred ' + str(e))
        tdt.SetTargetVal('RX8.gainError', 1)
        fontColor = (128, 128, 255)

    logging.info("Writing frames")

    ###############################
    while(True):

        # Read in image, convert to compatible data format for saving
        frames = pipeline.wait_for_frames()

        tdt_sample = round(tdt.GetTargetVal('RX8.zTime'))   # Do this as close as possible to the time the frame is returned
        frame_count = frames.frame_number
        time_stamp = frames.timestamp

        ir_frame_1 = frames.get_infrared_frame(1)
        ir_frame_2 = frames.get_infrared_frame(2)

        ir_image_1 = np.asanyarray(ir_frame_1.get_data())
        ir_image_2 = np.asanyarray(ir_frame_2.get_data())

        ir_image_1 = cv2.cvtColor(ir_image_1, cv2.COLOR_GRAY2BGR)
        ir_image_2 = cv2.cvtColor(ir_image_2, cv2.COLOR_GRAY2BGR)

        # Crop frame from camera 1
        shift_x = 150
        shift_y = 350
        crop_frame = ir_image_1[shift_x:shift_x + 360, shift_y:shift_y + 640]

        # Resize frame from camera 2
        rsz_frame = cv2.resize(ir_image_2, (640, 360))

        # Combine images
        ir_image_all = np.concatenate((crop_frame, rsz_frame), axis=1)

        center_val = get_center_value( ir_image_2, pts)

        # Increment counter and label frame
        add_text(cv2, ir_image_all, (1200, 20), center_val)
        add_text(cv2, ir_image_all, (650, 20), tdt_sample)
        add_text(cv2, ir_image_all, (1200, 350), time_stamp)
        add_text(cv2, ir_image_all, (650, 350), frame_count)

        # Write the frame to disk
        out.write(ir_image_all)

        # Write the time to text file
        fid.write('%d\t' % frame_count)
        fid.write('%d\t' % time_stamp)
        fid.write('%d\t' % tdt_sample)
        fid.write('%.3f\n' % center_val)

        # Show the user
        cv2.imshow('RealSense', ir_image_all)

        # Escape once tdt closes
        if tdt.GetSysMode() == 0:
            break

    # Close text file
    fid.close()

    # Release everything if job is finished
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()
    tdt.CloseConnection()

if __name__ == '__main__':
    main()
