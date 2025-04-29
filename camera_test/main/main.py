import time
import json
from camera_manager_instantarray import CameraManager
from serial_control import SerialController
from ball_detection import BallDetector
from visualization import FrameVisualizer
import numpy as np
from ball_triangulation_cv import BallTriangulatorP
# from ball_triangulation import BallTriangulator
from scipy.io import loadmat
from plot_3D import plot_3d


with open('hsv_values.json', 'r') as f:
    hsv_values = json.load(f)

stereo_params = loadmat('src/stereoParameters.mat')
pose_params = np.load('src/pose_parameters.npz')

def main():
    # Initialize components
    serial_controller = SerialController(port='/dev/cu.usbserial-0001', baudrate=115200)
    camera_manager = CameraManager(exposure_time=6000, gain=15, mode=0)
    ball_detector = BallDetector(max_workers=2, 
                                lower_limit=np.array([hsv_values['h_min'], hsv_values['s_min'], hsv_values['v_min']]), 
                                upper_limit=np.array([hsv_values['h_max'], hsv_values['s_max'], hsv_values['v_max']]),
                                downsample_ratio=0.5)  # Use 50% resolution for faster processing
    visualizer = FrameVisualizer(scale_factor=0.5)
    ball_triangulator_cv = BallTriangulatorP(stereo_params, pose_params, coord_frame='table')
    # ball_triangulator = BallTriangulator(stereo_params, pose_params)

    position_3D = []

    try:
        print("Press 'q' to quit")
        serial_controller.start_trigger()
        
        while True:
            # Get synchronized frames and measure processing time
            frame_start_time = time.time()
            img1, img2, bid, tms = camera_manager.get_synchronized_frames()

            # Process frames if available
            if img1 is not None and img2 is not None:
                # Detect balls in both frames
                img1, img2 = ball_detector.process_stereo_frames(img1, img2)

                # Triangulate ball position
                result = ball_triangulator_cv.process_frames_thread(img1[1], img2[1], tms)

                if result is not None:
                    position_3D.append(result)

                print("--------------------------------")
                
                # Calculate processing time
                frame_processing_time = (time.time() - frame_start_time) * 1000  # ms
                
                # # Display frames with stats
                # key = visualizer.display_frames(img1, img2, frame_processing_time)
                
                # # Check for quit key
                # if key == ord('q'):
                #     break
            else:
                # Handle synchronization errors
                print("No synchronized frames available")

    finally:
        # Clean up resources
        serial_controller.stop_trigger()
        camera_manager.close()
        ball_detector.close()
        visualizer.close()
        print("All resources released")

        position_3D = np.array(position_3D)
        np.savez('/Users/phuditthaksinawong/clone_github_table_tennis/table_tennis_robot/camera_test/data/data_paddle.npz', position_3D=position_3D)
        print(f"Max z val: {np.max(-position_3D[:,2])}")
        print(f"Min z val: {np.min(-position_3D[:,2])}")
        plot_3d( position_3D[:,1], position_3D[:,0], -position_3D[:,2])

        

if __name__ == "__main__":
    main() 