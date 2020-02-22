"""
Function to check realsense camera works
- No saving data to disk
- Press 'q' to close window

    Updated (ST) - 22nd Feb 2020 to include timestamps
"""
import pyrealsense2 as rs
import numpy as np
import cv2


def add_text(cv2, im, pos, prefix, value, color=(255, 255, 255)):

    txt = prefix + ': ' + str(round(value))
    cv2.putText(im, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():

    # Options
    fps = 30        # frames per second
    ir_num = 1      # IR camera (1 or 2)
    im_width = 1280
    im_height = 720

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, ir_num, im_width, im_height, rs.format.y8, fps)

    # Start streaming
    profile = pipeline.start(config)

    # Disable IR emitter
    sensor = profile.get_device().first_depth_sensor()
    sensor.set_option(rs.option.emitter_enabled, 0)

    while(True):

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # back_ts = frames.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)
        frame_ts = frames.get_frame_metadata(rs.frame_metadata_value.frame_timestamp)
        sensor_ts = frames.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)
        arrival_ts = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)

        ir_frame = frames.get_infrared_frame()
        ir_image = np.asanyarray(ir_frame.get_data())

        add_text(cv2, ir_image, (10, 610), 'Frame', frames.frame_number)
        add_text(cv2, ir_image, (10, 630), 'Timestamp', frames.timestamp)
        # add_text(cv2, ir_image, (10, 650), 'Backend', back_ts)
        add_text(cv2, ir_image, (10, 670), 'Frame', frame_ts)
        add_text(cv2, ir_image, (10, 690), 'Sensor', sensor_ts)
        add_text(cv2, ir_image, (10, 710), 'Arrival', arrival_ts)

        # Display the resulting frame
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', ir_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
