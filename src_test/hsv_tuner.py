import argparse
import time
from vision_utils import BaslerCamera, hsv_tuning_mode

def main():
    parser = argparse.ArgumentParser(description='Interactive HSV tuning for ball detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--name', type=str, default='Camera', help='Camera name (default: Camera)')
    parser.add_argument('--exposure', type=int, default=6000, help='Camera exposure time (default: 6000)')
    parser.add_argument('--gain', type=int, default=0, help='Camera gain (default: 0)')
    args = parser.parse_args()
    
    print(f"Initializing camera {args.camera} with exposure={args.exposure}, gain={args.gain}")
    
    # Initialize camera
    camera = BaslerCamera(
        camera_index=args.camera,
        exposure_time=args.exposure, 
        gain=args.gain
    )
    
    # Start camera thread
    camera.start()
    print("Waiting for camera to initialize...")
    time.sleep(2)
    
    # Run HSV tuning mode
    try:
        print(f"Starting HSV tuning mode for {args.name}")
        print("Instructions:")
        print("1. Adjust the HSV sliders to isolate the ball in the mask")
        print("2. Press 'p' to print current HSV values to the console")
        print("3. Press 'q' to quit")
        
        hsv_tuning_mode(camera, args.name)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Stopping camera...")
        camera.stop()
        camera.join()
        print("Done")

if __name__ == "__main__":
    main() 