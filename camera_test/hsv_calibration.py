import threading
import cv2
import numpy as np
import time
import json
from camera_manager_instantarray import CameraManager
from serial_control import SerialController

serial_controller = SerialController(port='COM4', baudrate=115200)
serial_controller.start_trigger()

# Global variables for color picker
selected_color = None
mouse_x, mouse_y = 0, 0
clicked_points = []  # Store clicked points

def nothing(x):
    pass

def mouse_callback(event, x, y, flags, param):
    global selected_color, mouse_x, mouse_y, clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale back from display coordinates to original frame coordinates
        # Since display is scaled to 40%, multiply by 1/0.4 = 2.5
        original_x = int(x * 2.5)
        original_y = int(y * 2.5)
        
        mouse_x, mouse_y = original_x, original_y
        frame = param
        
        # Add clicked point to the list (store in original frame coordinates)
        clicked_points.append((original_x, original_y))
        
        if frame is not None:
            # Get the color at the clicked position (using original coordinates)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Ensure coordinates are within frame bounds
            h, w = hsv.shape[:2]
            original_y = min(original_y, h-1)
            original_x = min(original_x, w-1)
            selected_color = hsv[original_y, original_x]
            print(f"Selected HSV color: {selected_color}")
            
            # Set HSV ranges based on selected color
            h, s, v = selected_color
            
            # Set reasonable ranges around the selected color
            h_range = 10  # Hue range
            s_range = 40  # Saturation range
            v_range = 40  # Value range
            
            # Calculate min and max values
            h_min = max(0, h - h_range)
            h_max = min(179, h + h_range)
            s_min = max(0, s - s_range)
            s_max = min(255, s + s_range)
            v_min = max(0, v - v_range)
            v_max = min(255, v + v_range)
            
            # Update trackbars
            cv2.setTrackbarPos('H min', 'HSV Trackbars', h_min)
            cv2.setTrackbarPos('H max', 'HSV Trackbars', h_max)
            cv2.setTrackbarPos('S min', 'HSV Trackbars', s_min)
            cv2.setTrackbarPos('S max', 'HSV Trackbars', s_max)
            cv2.setTrackbarPos('V min', 'HSV Trackbars', v_min)
            cv2.setTrackbarPos('V max', 'HSV Trackbars', v_max)
            
            print(f"Set HSV ranges:")
            print(f"H: {h_min}-{h_max}")
            print(f"S: {s_min}-{s_max}")
            print(f"V: {v_min}-{v_max}")

def create_color_preview(hsv_value):
    """Create a preview image of the selected HSV color"""
    preview = np.zeros((100, 100, 3), dtype=np.uint8)
    preview[:, :] = hsv_value
    preview = cv2.cvtColor(preview, cv2.COLOR_HSV2BGR)
    return preview

def save_hsv_values(h_min, h_max, s_min, s_max, v_min, v_max, filename='hsv_values.json'):
    """Save HSV values to a JSON file"""
    values = {
        'h_min': h_min,
        'h_max': h_max,
        's_min': s_min,
        's_max': s_max,
        'v_min': v_min,
        'v_max': v_max
    }
    with open(filename, 'w') as f:
        json.dump(values, f)
    print(f"HSV values saved to {filename}")

def load_hsv_values(filename='hsv_values.json'):
    """Load HSV values from a JSON file"""
    try:
        with open(filename, 'r') as f:
            values = json.load(f)
        return values
    except FileNotFoundError:
        return None

def main():
    try:
        global clicked_points  # Add global declaration to access clicked_points
        
        # Create windows for trackbars and preview
        cv2.namedWindow('HSV Trackbars')
        cv2.resizeWindow('HSV Trackbars', 400, 400)
        cv2.namedWindow('Color Preview')
        cv2.namedWindow('Camera 1 Original')
        cv2.namedWindow('Camera 2 Original')
        
        # Create trackbars
        cv2.createTrackbar('H min', 'HSV Trackbars', 0, 179, nothing)
        cv2.createTrackbar('H max', 'HSV Trackbars', 179, 179, nothing)
        cv2.createTrackbar('S min', 'HSV Trackbars', 0, 255, nothing)
        cv2.createTrackbar('S max', 'HSV Trackbars', 255, 255, nothing)
        cv2.createTrackbar('V min', 'HSV Trackbars', 0, 255, nothing)
        cv2.createTrackbar('V max', 'HSV Trackbars', 255, 255, nothing)
        
        # Create and start stereo camera system
        camera_manager = CameraManager(gain=20,mode=0)
        
        # Initialize serial controller and start trigger AFTER camera setup
        
        
        
        print("Cameras initialized and started successfully")
        print("Instructions:")
        print("1. Click on the object in the camera view to sample its color")
        print("2. Use trackbars to fine-tune the HSV range")
        print("3. Press 's' to save current HSV values")
        print("4. Press 'l' to load saved HSV values")
        print("5. Press 'c' to clear all clicked points")
        print("6. Press 'q' to quit")
        
        # Load saved HSV values if they exist
        saved_values = load_hsv_values()
        if saved_values:
            cv2.setTrackbarPos('H min', 'HSV Trackbars', saved_values['h_min'])
            cv2.setTrackbarPos('H max', 'HSV Trackbars', saved_values['h_max'])
            cv2.setTrackbarPos('S min', 'HSV Trackbars', saved_values['s_min'])
            cv2.setTrackbarPos('S max', 'HSV Trackbars', saved_values['s_max'])
            cv2.setTrackbarPos('V min', 'HSV Trackbars', saved_values['v_min'])
            cv2.setTrackbarPos('V max', 'HSV Trackbars', saved_values['v_max'])

        
        while True:
            # Get frames from stereo system
            frame1, frame2, block_id1, block_id2 = camera_manager.get_synchronized_frames()
            
            if frame1 is not None and frame2 is not None:
                # Set mouse callback for color picking
                cv2.setMouseCallback('Camera 1 Original', mouse_callback, frame1)
                
                # Convert to HSV
                hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
                hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
                
                # Get trackbar values
                h_min = cv2.getTrackbarPos('H min', 'HSV Trackbars')
                h_max = cv2.getTrackbarPos('H max', 'HSV Trackbars')
                s_min = cv2.getTrackbarPos('S min', 'HSV Trackbars')
                s_max = cv2.getTrackbarPos('S max', 'HSV Trackbars')
                v_min = cv2.getTrackbarPos('V min', 'HSV Trackbars')
                v_max = cv2.getTrackbarPos('V max', 'HSV Trackbars')
                
                # Create mask with current HSV values
                lower = np.array([h_min, s_min, v_min])
                upper = np.array([h_max, s_max, v_max])

                hsv1 = cv2.GaussianBlur(hsv1, (7, 7), 0)
                hsv2 = cv2.GaussianBlur(hsv2, (7, 7), 0)
                
                # Apply mask and preprocessing
                mask1 = cv2.inRange(hsv1, lower, upper)
                mask2 = cv2.inRange(hsv2, lower, upper)
                
                # Apply morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
                mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
                
                # Find and draw contours
                contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                frame1_with_contours = frame1.copy()
                frame2_with_contours = frame2.copy()
                
                # Draw contours that match ball size criteria
                for contour in contours1:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame1_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                for contour in contours2:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame2_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw dots at all clicked points
                for point in clicked_points:
                    # Scale point coordinates based on the display scale factor
                    px, py = point
                    cv2.circle(frame1_with_contours, (px, py), 5, (0, 0, 255), -1)  # Draw a red dot
                
                # Scale frames to 40%
                frame1_scaled = cv2.resize(frame1_with_contours, None, fx=0.4, fy=0.4)
                mask1_scaled = cv2.resize(mask1, None, fx=0.4, fy=0.4)
                frame2_scaled = cv2.resize(frame2_with_contours, None, fx=0.4, fy=0.4)
                mask2_scaled = cv2.resize(mask2, None, fx=0.4, fy=0.4)
                
                # Display results
                cv2.imshow('Camera 1 Original', frame1_scaled)
                cv2.imshow('Camera 1 Mask', mask1_scaled)
                cv2.imshow('Camera 2 Original', frame2_scaled)
                cv2.imshow('Camera 2 Mask', mask2_scaled)
                
                # Show color preview if a color is selected
                if selected_color is not None:
                    preview = create_color_preview(selected_color)
                    cv2.imshow('Color Preview', preview)
                
                # Print current HSV values
                print(f"\rCurrent HSV values - Lower: [{h_min}, {s_min}, {v_min}], Upper: [{h_max}, {s_max}, {v_max}]", end='')
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    save_hsv_values(h_min, h_max, s_min, s_max, v_min, v_max)
                elif key == ord('l'):
                    saved_values = load_hsv_values()
                    if saved_values:
                        cv2.setTrackbarPos('H min', 'HSV Trackbars', saved_values['h_min'])
                        cv2.setTrackbarPos('H max', 'HSV Trackbars', saved_values['h_max'])
                        cv2.setTrackbarPos('S min', 'HSV Trackbars', saved_values['s_min'])
                        cv2.setTrackbarPos('S max', 'HSV Trackbars', saved_values['s_max'])
                        cv2.setTrackbarPos('V min', 'HSV Trackbars', saved_values['v_min'])
                        cv2.setTrackbarPos('V max', 'HSV Trackbars', saved_values['v_max'])
                elif key == ord('c'):
                    # Clear all clicked points
                    clicked_points = []
                    print("\nCleared all clicked points")

            else:
                print("No synchronized frames available")
                print(f"Block ID 1: {block_id1} != Block ID 2: {block_id2}")

    except KeyboardInterrupt:
        print("\nStopping cameras...")
    finally:
        camera_manager.close()
        serial_controller.stop_trigger()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
