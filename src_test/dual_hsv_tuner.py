import argparse
import time
import cv2
import numpy as np
from threading import Thread, Event
from vision_utils import BaslerCamera, create_hsv_trackbar_window

def camera_thread(camera, get_hsv_values, display_name, stop_event):
    """Thread function to process frames from a camera with interactive HSV adjustment"""
    while not stop_event.is_set():
        frame = camera.get_frame()
        
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Get current HSV thresholds
        lower_limit, upper_limit = get_hsv_values()
        
        # Process the frame
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred_image = cv2.GaussianBlur(hsv_image, (7, 7), 0)
        mask = cv2.inRange(blurred_image, lower_limit, upper_limit)
        
        # Apply morphological operations to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        # Find contours for visualization
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Make a copy for visualization
        result_image = frame.copy()
        
        # Draw all contours
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 1)
        
        # Resize images for display
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        result_image = cv2.resize(result_image, (new_width, new_height))
        mask_display = cv2.resize(mask, (new_width, new_height))
        mask_colored = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
        
        # Display original with contours and mask
        cv2.imshow(f"{display_name} Original", result_image)
        cv2.imshow(f"{display_name} Mask", mask_colored)

def main():
    parser = argparse.ArgumentParser(description='Interactive HSV tuning for stereo cameras')
    parser.add_argument('--left-camera', type=int, default=0, help='Left camera index (default: 0)')
    parser.add_argument('--right-camera', type=int, default=1, help='Right camera index (default: 1)')
    parser.add_argument('--exposure', type=int, default=6000, help='Camera exposure time (default: 6000)')
    parser.add_argument('--gain', type=int, default=0, help='Camera gain (default: 0)')
    args = parser.parse_args()
    
    print(f"Initializing cameras with exposure={args.exposure}, gain={args.gain}")
    
    # Initialize cameras
    left_camera = BaslerCamera(
        camera_index=args.left_camera,
        exposure_time=args.exposure, 
        gain=args.gain
    )
    
    right_camera = BaslerCamera(
        camera_index=args.right_camera,
        exposure_time=args.exposure, 
        gain=args.gain
    )
    
    # Start camera threads
    left_camera.start()
    right_camera.start()
    print("Waiting for cameras to initialize...")
    time.sleep(2)
    
    # Create trackbar windows for both cameras
    left_get_hsv = create_hsv_trackbar_window("Left Camera")
    right_get_hsv = create_hsv_trackbar_window("Right Camera")
    
    # Create an event for stopping the threads
    stop_event = Event()
    
    # Create threads for processing camera frames
    left_process_thread = Thread(
        target=camera_thread, 
        args=(left_camera, left_get_hsv, "Left Camera", stop_event)
    )
    
    right_process_thread = Thread(
        target=camera_thread, 
        args=(right_camera, right_get_hsv, "Right Camera", stop_event)
    )
    
    # Start processing threads
    left_process_thread.start()
    right_process_thread.start()
    
    # Run HSV tuning mode
    try:
        print("Starting dual HSV tuning mode")
        print("Instructions:")
        print("1. Adjust the HSV sliders to isolate the ball in each camera")
        print("2. Press 'p' to print current HSV values for both cameras")
        print("3. Press 'q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                left_lower, left_upper = left_get_hsv()
                right_lower, right_upper = right_get_hsv()
                
                print("\nHSV Values:")
                print(f"Left camera lower limit: {left_lower}")
                print(f"Left camera upper limit: {left_upper}")
                print(f"Right camera lower limit: {right_lower}")
                print(f"Right camera upper limit: {right_upper}")
                print("Add these values to your ball_tracker_basler.py script:")
                print(f"left_color_limits = (np.array({list(left_lower)}), np.array({list(left_upper)}))")
                print(f"right_color_limits = (np.array({list(right_lower)}), np.array({list(right_upper)}))")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stop and clean up
        print("Stopping...")
        stop_event.set()
        
        left_process_thread.join()
        right_process_thread.join()
        
        left_camera.stop()
        right_camera.stop()
        
        left_camera.join()
        right_camera.join()
        
        cv2.destroyAllWindows()
        print("Done")

if __name__ == "__main__":
    main() 