import time
import json
from camera_manager import CameraManager
from serial_control import SerialController
from ball_detection import BallDetector
from visualization import FrameVisualizer
import numpy as np

with open('hsv_values.json', 'r') as f:
    hsv_values = json.load(f)

def main():
    # Initialize components
    serial_controller = SerialController(port='COM5', baudrate=115200)
    camera_manager = CameraManager(exposure_time=6000, gain=20)
    ball_detector = BallDetector(max_workers=2, lower_limit=np.array([hsv_values['h_min'], hsv_values['s_min'], hsv_values['v_min']]), upper_limit=np.array([hsv_values['h_max'], hsv_values['s_max'], hsv_values['v_max']]))
    visualizer = FrameVisualizer(scale_factor=0.5)
    
    count = 0  # Counter for synchronization errors
    
    try:
        print("Press 'q' to quit")
        serial_controller.start_trigger()
        
        while True:
            # Get synchronized frames and measure processing time
            frame_start_time = time.time()
            img1, img2, block_id1, block_id2 = camera_manager.get_synchronized_frames()

            # Process frames if available
            if img1 is not None and img2 is not None:
                # Detect balls in both frames
                img1, img2 = ball_detector.process_stereo_frames(img1, img2)
                
                # Calculate processing time
                frame_processing_time = (time.time() - frame_start_time) * 1000  # ms
                
                # Display frames with stats
                key = visualizer.display_frames(img1, img2, frame_processing_time)
                
                # Check for quit key
                if key == ord('q'):
                    break
            else:
                # Handle synchronization errors
                print("No synchronized frames available")
                print(f"Block ID 1: {block_id1} != Block ID 2: {block_id2}")
                count += 1
                print(f"Count: {count}")
                

                
    finally:
        # Clean up resources
        serial_controller.stop_trigger()
        camera_manager.close()
        ball_detector.close()
        visualizer.close()
        print("All resources released")

if __name__ == "__main__":
    main() 