"""
Function to check realsense camera works
- No saving data to disk
- Press 'q' to close window
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import sys

sys.path.insert(0, 'C:/Users/Dumbo/Documents/Python')
# from realsense_device_manager import DeviceManager

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# device_manager.load_settings_json("C:/Users/Dumbo/Documents/Python/IR_Settings.json")

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Disable IR emitter
sensor = profile.get_device().first_depth_sensor()
sensor.set_option(rs.option.emitter_enabled, 0)

# Set exposure
# s = profile.get_device().query_sensors()[1]
# s.set_option(rs.option.enable_auto_exposure, 1)
# s.set_option(rs.option.exposure, 100000)
# s.set_option(rs.option.gain, 30)

# Create function for adjusting image gamma
# def adjust_gamma(image, gamma=1.0):
#     # build a lookup table mapping the pixel values [0, 255] to
#     # their adjusted gamma values
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#         for i in np.arange(0, 256)]).astype("uint8")

#     # apply gamma correction using the lookup table
#     return cv2.LUT(image, table)




while(True):

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    # depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()
    ir_frame = frames.get_infrared_frame()

    # Convert images to numpy arrays
    # depth_image = np.asanyarray(depth_frame.get_data())
    # color_image = np.asanyarray(color_frame.get_data())
    ir_image = np.asanyarray(ir_frame.get_data())





    # Display the resulting frame
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', ir_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
pipeline.stop()
cv2.destroyAllWindows()

