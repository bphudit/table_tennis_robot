import cv2
import pyrealsense2 as rs
import numpy as np
from src.intel import init_cam_color
# Define pipeline for each camera
pipeline_right = rs.pipeline()
pipeline_left = rs.pipeline()

# Configure both pipelines to connect to specific cameras
config_right = rs.config()
config_left = rs.config()

serial_number_right, serial_number_left = ['152222070695', '153122073043']

# Add serial numbers for each camera
config_right.enable_device(serial_number_right)
config_left.enable_device(serial_number_left)

# Enable color streams
config_right.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config_left.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start both pipelines
pipeline_left.start(config_left)
pipeline_right.start(config_right)

# pipeline_left, pipeline_right = init_cam_color()

num = 0

while True:

    frames_left = pipeline_left.wait_for_frames()
    frames_right = pipeline_right.wait_for_frames()

    # Get color frames
    color_frame_left = frames_left.get_color_frame()
    color_frame_right = frames_right.get_color_frame()

    # Convert RealSense frames to numpy arrays
    color_image_left = np.asanyarray(color_frame_left.get_data())
    color_image_right = np.asanyarray(color_frame_right.get_data())

    # Get the current frame size
    height, width, _ = color_image_left.shape

    # # Resize the frame
    scale_percent = 80
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    dim = (new_width, new_height)
    resized_left = cv2.resize(color_image_left, dim, interpolation = cv2.INTER_AREA)
    resized_right = cv2.resize(color_image_right, dim, interpolation = cv2.INTER_AREA)


    cv2.imshow('Left Camera', resized_left)
    cv2.imshow('Right Camera', resized_right)

    # cv2.imshow('Left Camera', color_image_left)
    # cv2.imshow('Right Camera ', color_image_right)

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('right/img' + str(num) + '.png', color_image_right)
        cv2.imwrite('left/img' + str(num) + '.png', color_image_left)
        print("image saved!")
        num += 1
    elif k == ord('q'):
        break

pipeline_left.stop()
pipeline_right.stop()

cv2.destroyAllWindows()