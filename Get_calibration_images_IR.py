import pyrealsense2 as rs
import numpy as np
import time
import cv2


def configure_camera(ir_num, im_width, im_height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, ir_num, im_width, im_height, rs.format.y8, fps)

    profile = pipeline.start(config)    # Start streaming

    sensor = profile.get_device().first_depth_sensor()  # Disable IR emitter
    sensor.set_option(rs.option.emitter_enabled, 0)

    s = profile.get_device().query_sensors()[0]     # Set exposure
    s.set_option(rs.option.enable_auto_exposure, 1)

    return pipeline


def get_frame(pipeline):

    frames = pipeline.wait_for_frames()
    ir_frame = frames.get_infrared_frame()
    ir_image = np.asanyarray(ir_frame.get_data())
    ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

    return ir_image


def save_image(save_dir, camera_name, ir_image):

    file_name = time.strftime('%Y-%m-%d_IR_%H-%M-%S.png')
    file_name = file_name.replace('IR', camera_name)
    save_path = save_dir + file_name

    print(file_name)

    cv2.imwrite(save_path, ir_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def main(ir_num):

    fps = 30    # frames per second
    im_width = 1280
    im_height = 720
    save_dir = 'C:/Users/Dumbo/Pictures/Jumbo_calibration/'

    pipeline = configure_camera(ir_num, im_width, im_height, fps)

    time.sleep(5)   # Wait for sensor to adapt

    image = get_frame(pipeline)

    save_image(save_dir, 'IR' + str(ir_num), image)

    pipeline.stop()


if __name__ == '__main__':
    main(1)
    main(2)


