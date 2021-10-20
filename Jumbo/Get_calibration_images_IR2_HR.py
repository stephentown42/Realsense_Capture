import pyrealsense2 as rs
import numpy as np
import time
import cv2

# Options
fps = 30    # frames per second
im_width = 1280
im_height = 720

# Configure realsense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 2, im_width, im_height, rs.format.y8, fps)

# Start streaming
profile = pipeline.start(config)

# Disable IR emitter
sensor = profile.get_device().first_depth_sensor()
sensor.set_option(rs.option.emitter_enabled, 0)

# Set exposure
s = profile.get_device().query_sensors()[0]
s.set_option(rs.option.enable_auto_exposure, 1)
# s.set_option(rs.option.exposure, 120000)

# # Wait for sensor to adapt
time.sleep(5)

# for x in range(0, 3):
    # input("Press Enter to continue...")

# Get example frame
frames = pipeline.wait_for_frames()
ir_frame = frames.get_infrared_frame()
ir_image = np.asanyarray(ir_frame.get_data())
ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)


# Define file name based on current time
im_name_1 = time.strftime('%Y-%m-%d_IR1_%H-%M-%S.png')
im_name_2 = time.strftime('%Y-%m-%d_IR2_%H-%M-%S.png')

print(im_name_1)

save_path = 'C:/Users/Dumbo/Pictures/Jumbo_calibration/' + im_name_1
save_path = 'C:/Users/Dumbo/Pictures/Jumbo_calibration/' + im_name_2

# save image with lower compression—bigger file size but faster decoding
cv2.imwrite(save_path, ir_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Release everything if job is finished
pipeline.stop()
