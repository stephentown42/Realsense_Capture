'''
Streams video from an Infra-red camera on a specific Realsense Camera to a file on disk

TODO:
    - Remove dependency on camera serial number 
    - Update CV2 to resolve compatability with H264 codec (much newer versions available for both systems)

Version History
    2019-07-15: Created by Stephen Town (ST)
    2021-10-19: Updated by ST
'''

import os

import pyrealsense2 as rs
import numpy as np
import cv2
import time


class recording():

    def __init__(self, fps, width, height, serial, sync_mode) -> None:

        self.fps = fps
        self.width = width
        self.height = height
        self.serial = serial
        self.sync_mode = sync_mode


    def add_pipeline(self) -> None:
        """ Creates pipeline through which data acquisition passes"""
        self.pipeline = rs.pipeline()


    def create_config(self, ir1: bool, ir2: bool) -> None:
        """ Creates configuration, through which sensors are then enabled"""

        self.config = rs.config()
        self.config.enable_device(self.serial) 
        
        if ir1:
            self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
        
        if ir2:
            self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)


    def start(self, record: bool, rec_path: str) -> None:

        # Create video writer object if recording
        if record:
            file_name = tiqme.strftime('%Y-%m-%dT%H-%M-%S_rs.avi')
            file_path = os.path.join( rec_path, file_name)

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.vid_obj = cv2.VideoWriter(file_path, fourcc, self.fps, (self.width, self.height))

        # Start Camera
        self.profile = self.pipeline.start(self.config)
        print(f"Starting camera (Serial no: {self.serial}) - press 'q' to stop")


    def get_image(self) -> None:
        """ Get image from camera"""

        # Get frame from camera
        frames = self.pipeline.wait_for_frames()
        ir_frame = frames.get_infrared_frame()
        
        # Format for visualization
        ir_image = np.asanyarray(ir_frame.get_data())    
        ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

        # Add to video if recording
        if hasattr(self, 'vid_obj'):
            self.vid_obj.write(ir_image)    

        return ir_image
    

    def stop(self) -> None:
        """ Release camera and close video file"""

        # Close pipeline from camera
        print('Recording stopping')
        self.pipeline.stop()

        # Stop video if recording
        cv2.destroyAllWindows()
        if hasattr(self, 'vid_obj'):
            self.vid_obj.release()


def main():

    # Create recording with these settings
    rec = recording(30, 640, 360, '831612073855', 1)              #<= Low Res, small Files 
    #rec = recording(30, 1280, 720, '831612073855', 1)             #<= HIGH Res, Big Files

    rec.add_pipeline()
    rec.create_config(ir1=True, ir2=False)

    # Start camera and get image
    rec.start(record=True, rec_path='C:/Users/Squid/Videos/SyncTest/')

    while(True):

        im = rec.get_image()

        # Display the resulting frames
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close pipeline
    rec.stop()


if __name__ == "__main__":
    main()
