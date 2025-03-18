import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread, Event
from pypylon import pylon

class BaslerCamera(Thread):
    """
    Thread-based Basler camera handler for continuous frame capturing.
    """
    def __init__(self, camera_index, exposure_time=10000, gain=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.current_frame = None
        self.camera = None
        self.exposure_time = exposure_time
        self.gain = gain
        self.converter = pylon.ImageFormatConverter()
        
        # Set up format converter
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
    def initialize_camera(self):
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            if not devices:
                raise Exception("No camera found!")
            
            if self.camera_index >= len(devices):
                raise Exception(f"Camera index {self.camera_index} is out of range!")
            
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[self.camera_index]))
            self.camera.Open()
            
            self.camera.ExposureTime.SetValue(self.exposure_time)
            self.camera.Gain.SetValue(self.gain)
            
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print(f"Camera {self.camera_index} initialized successfully")
            
        except Exception as e:
            print(f"Error initializing camera {self.camera_index}: {str(e)}")
            raise
    
    def run(self):
        self.running = True
        self.initialize_camera()
        
        while self.running:
            if self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    image = self.converter.Convert(grab_result)
                    self.current_frame = image.GetArray()
                
                grab_result.Release()
    
    def stop(self):
        self.running = False
        if self.camera:
            self.camera.Close()
    
    def get_frame(self):
        return self.current_frame

class BallTracker(Thread):
    """
    Thread to process camera frames and detect the ball.
    """
    def __init__(self, camera, intrinsic_inv, output_queue, val_queue, 
                 lower_limit, upper_limit, stop_event, min_area=0, max_area=1000000, mask_queue=None):
        super().__init__()
        self.camera = camera
        self.intrinsic_inv = intrinsic_inv
        self.output_queue = output_queue
        self.val_queue = val_queue
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.stop_event = stop_event
        self.min_area = min_area
        self.max_area = max_area
        self.mask_queue = mask_queue  # Queue for the HSV mask
    
    def run(self):
        try:
            while not self.stop_event.is_set():
                # Get the latest frame
                color_image = self.camera.get_frame()
                
                if color_image is None:
                    time.sleep(0.01)  # Short sleep to prevent CPU spinning
                    continue
                
                # Process the frame to detect the ball
                u, v, processed_image, mask = self.detect_ball(color_image)
                
                # If ball is detected, compute its position in camera coordinates
                if u is not None and v is not None:
                    if self.val_queue.full():
                        self.val_queue.get()
                    x = self.intrinsic_inv @ np.array([u, v, 1])
                    self.val_queue.put(x)
                
                # Put the processed frame into the output queue for display
                if self.output_queue.full():
                    self.output_queue.get()
                self.output_queue.put(processed_image)
                
                # If mask queue is provided, put the mask into it
                if self.mask_queue is not None:
                    if self.mask_queue.full():
                        self.mask_queue.get()
                    # Convert mask to BGR for visualization
                    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    self.mask_queue.put(mask_colored)
                
        except Exception as e:
            print(f"Error in ball tracker: {e}")
    
    def detect_ball(self, color_image):
        """
        Detect the ball in the given image using color thresholding.
        
        Returns:
            u, v: Center coordinates of the detected ball (or None if not found)
            processed_image: Image with detection visualization
            mask: Binary mask showing the thresholded areas
        """
        u, v = None, None
        
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(hsv_image, (7, 7), 0)
        
        # Create mask using color thresholds
        mask = cv2.inRange(blurred_image, self.lower_limit, self.upper_limit)
        
        # Apply morphological operations to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Make a copy for visualization
        result_image = color_image.copy()
        
        # Find the ball contour based on area constraints
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                u, v = x + w // 2, y + h // 2
                
                # Draw rectangle around the detected ball
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                break
        
        return u, v, result_image, mask

def init_stereo_cameras(left_camera_index=0, right_camera_index=1, 
                       left_color_limits=None, right_color_limits=None, 
                       exposure_time=10000, gain=0):
    """
    Initialize stereo Basler cameras for ball tracking.
    
    Args:
        left_camera_index: Index of the left camera
        right_camera_index: Index of the right camera
        left_color_limits: Tuple of (lower_limit, upper_limit) for left camera
        right_color_limits: Tuple of (lower_limit, upper_limit) for right camera
        exposure_time: Camera exposure time
        gain: Camera gain
        
    Returns:
        Camera objects, queues for processed images and detected positions,
        and threading control objects
    """
    # Default color limits if none provided
    if left_color_limits is None:
        left_color_limits = (np.array([0, 212, 141]), np.array([81, 255, 255]))
    
    if right_color_limits is None:
        right_color_limits = (np.array([0, 109, 151]), np.array([81, 255, 255]))
    
    # Extract limits
    lower_limit_left, upper_limit_left = left_color_limits
    lower_limit_right, upper_limit_right = right_color_limits
    
    # Initialize cameras
    left_camera = BaslerCamera(left_camera_index, exposure_time, gain)
    right_camera = BaslerCamera(right_camera_index, exposure_time, gain)
    
    # Start camera threads
    left_camera.start()
    right_camera.start()
    
    # Wait for cameras to initialize
    time.sleep(2)
    
    # Create queues for processed frames and detected positions
    left_frame_queue = Queue(maxsize=1)
    right_frame_queue = Queue(maxsize=1)
    left_pos_queue = Queue(maxsize=1)
    right_pos_queue = Queue(maxsize=1)
    left_mask_queue = Queue(maxsize=1)
    right_mask_queue = Queue(maxsize=1)
    
    # Create stop event for graceful shutdown
    stop_event = Event()
    
    return (left_camera, right_camera, 
            left_frame_queue, right_frame_queue, 
            left_pos_queue, right_pos_queue, 
            left_mask_queue, right_mask_queue, 
            stop_event)

def start_ball_tracking(left_camera, right_camera, 
                      left_frame_queue, right_frame_queue, 
                      left_pos_queue, right_pos_queue, 
                      K1_inv, K2_inv, stop_event, 
                      left_mask_queue=None, right_mask_queue=None,
                      left_color_limits=None, right_color_limits=None):
    """
    Start ball tracking threads for both cameras.
    
    Args:
        left_camera, right_camera: Camera objects
        left_frame_queue, right_frame_queue: Queues for processed frames
        left_pos_queue, right_pos_queue: Queues for detected positions
        K1_inv, K2_inv: Inverse intrinsic matrices
        stop_event: Event to signal thread termination
        left_mask_queue, right_mask_queue: Queues for HSV masks
        left_color_limits, right_color_limits: Color thresholds for ball detection
        
    Returns:
        Tracker thread objects
    """
    # Default color limits if none provided
    if left_color_limits is None:
        left_color_limits = (np.array([0, 212, 141]), np.array([81, 255, 255]))
    
    if right_color_limits is None:
        right_color_limits = (np.array([0, 109, 151]), np.array([81, 255, 255]))
    
    # Extract limits
    lower_limit_left, upper_limit_left = left_color_limits
    lower_limit_right, upper_limit_right = right_color_limits
    
    # Create and start tracker threads
    left_tracker = BallTracker(
        left_camera, K1_inv, left_frame_queue, left_pos_queue, 
        lower_limit_left, upper_limit_left, stop_event, mask_queue=left_mask_queue
    )
    
    right_tracker = BallTracker(
        right_camera, K2_inv, right_frame_queue, right_pos_queue, 
        lower_limit_right, upper_limit_right, stop_event, mask_queue=right_mask_queue
    )
    
    left_tracker.start()
    right_tracker.start()
    
    return left_tracker, right_tracker

def calculate_3d_position(x1, x2, R, T, pose_rotation, pose_translation):
    """
    Calculate the 3D position of a point from stereo camera observations.
    
    Args:
        x1: Normalized coordinates from left camera (x, y, 1)
        x2: Normalized coordinates from right camera (x, y, 1)
        R: Rotation matrix between cameras
        T: Translation vector between cameras
        pose_rotation: Rotation matrix for coordinate transformation
        pose_translation: Translation vector for coordinate transformation
        
    Returns:
        3D position (x, y, z) in the world coordinate system
    """
    # Set up and solve for depths d1 and d2
    A = np.array([
        [x1[0], -R[0, 0] * x2[0] - R[0, 1] * x2[1] - R[0, 2]],
        [x1[1], -R[1, 0] * x2[0] - R[1, 1] * x2[1] - R[1, 2]],
        [1, -R[2, 0] * x2[0] - R[2, 1] * x2[1] - R[2, 2]]
    ])
    B = T.flatten()

    # Solve for d1 and d2
    depths = np.linalg.lstsq(A, B, rcond=None)[0]
    d1 = depths[0]

    # Compute the 3D point in Camera 1's frame
    X = d1 * x1
    
    # Transformation to table coordinate
    X = pose_rotation.T @ (X.reshape((3,1)) - pose_translation)
    
    return X

def cleanup_resources(left_camera, right_camera, left_tracker, right_tracker, stop_event):
    """
    Clean up resources by stopping all threads and closing cameras.
    """
    # Signal threads to stop
    stop_event.set()
    
    # Join tracker threads
    left_tracker.join()
    right_tracker.join()
    
    # Stop and close cameras
    left_camera.stop()
    right_camera.stop()
    
    # Join camera threads
    left_camera.join()
    right_camera.join()
    
    # Close any open windows
    cv2.destroyAllWindows()

def create_hsv_trackbar_window(camera_name="Camera"):
    """
    Create an interactive window with trackbars to adjust HSV thresholds for ball detection.
    
    Args:
        camera_name: Name to display in the window title
        
    Returns:
        Callback function to get the current HSV values
    """
    window_name = f"{camera_name} HSV Tuning"
    cv2.namedWindow(window_name)
    
    # Initial values
    h_min, s_min, v_min = 0, 100, 100
    h_max, s_max, v_max = 80, 255, 255
    
    # Create trackbars
    cv2.createTrackbar('H Min', window_name, h_min, 179, lambda x: None)
    cv2.createTrackbar('S Min', window_name, s_min, 255, lambda x: None)
    cv2.createTrackbar('V Min', window_name, v_min, 255, lambda x: None)
    cv2.createTrackbar('H Max', window_name, h_max, 179, lambda x: None)
    cv2.createTrackbar('S Max', window_name, s_max, 255, lambda x: None)
    cv2.createTrackbar('V Max', window_name, v_max, 255, lambda x: None)
    
    def get_hsv_values():
        """Get current values from trackbars"""
        h_min = cv2.getTrackbarPos('H Min', window_name)
        s_min = cv2.getTrackbarPos('S Min', window_name)
        v_min = cv2.getTrackbarPos('V Min', window_name)
        h_max = cv2.getTrackbarPos('H Max', window_name)
        s_max = cv2.getTrackbarPos('S Max', window_name)
        v_max = cv2.getTrackbarPos('V Max', window_name)
        
        lower_limit = np.array([h_min, s_min, v_min])
        upper_limit = np.array([h_max, s_max, v_max])
        
        return lower_limit, upper_limit
    
    return get_hsv_values

def hsv_tuning_mode(camera, camera_name="Camera"):
    """
    Run an interactive HSV tuning mode for a single camera.
    This is helpful for finding the right HSV thresholds for ball detection.
    
    Args:
        camera: Camera object that implements get_frame() method
        camera_name: Name to display in the window title
    """
    get_hsv_values = create_hsv_trackbar_window(camera_name)
    
    print(f"HSV tuning mode for {camera_name}")
    print("Press 'p' to print current HSV values")
    print("Press 'q' to quit")
    
    while True:
        frame = camera.get_frame()
        
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Get current HSV thresholds
        lower_limit, upper_limit = get_hsv_values()
        
        # Convert to HSV and apply mask
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
        scale = min(1.0, 800 / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        result_image = cv2.resize(result_image, (new_width, new_height))
        mask_display = cv2.resize(mask, (new_width, new_height))
        mask_colored = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
        
        # Display original with contours and mask
        cv2.imshow(f"{camera_name} Original", result_image)
        cv2.imshow(f"{camera_name} Mask", mask_colored)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"{camera_name} HSV values:")
            print(f"Lower limit: {lower_limit}")
            print(f"Upper limit: {upper_limit}")
    
    cv2.destroyAllWindows()

# Example usage:
"""
# Load calibration data
data = loadmat('stereoParameters.mat')
data2 = np.load('pose_parameters.npz')

# Extract calibration parameters
pose_rotation = data2['rotation_matrix']
pose_translation = data2['translation_matrix']
K1 = data['K1']  # Left camera intrinsic matrix
K2 = data['K2']  # Right camera intrinsic matrix
R = data['R']    # Rotation matrix between cameras
T = data['T'].reshape((3,1))  # Translation vector between cameras

# Prepare matrices for 3D reconstruction
R = np.linalg.inv(R)
T = -R @ T
K1_inv = np.linalg.inv(K1)
K2_inv = np.linalg.inv(K2)

# Initialize cameras
left_camera, right_camera, left_frame_queue, right_frame_queue, left_pos_queue, right_pos_queue, left_mask_queue, right_mask_queue, stop_event = init_stereo_cameras()

# Start ball tracking
left_tracker, right_tracker = start_ball_tracking(
    left_camera, right_camera, left_frame_queue, right_frame_queue, 
    left_pos_queue, right_pos_queue, K1_inv, K2_inv, stop_event, 
    left_mask_queue, right_mask_queue
)

try:
    while True:
        # Get processed frames
        if not left_frame_queue.empty():
            left_frame = left_frame_queue.get()
            cv2.imshow("Left Camera", left_frame)

        if not right_frame_queue.empty():
            right_frame = right_frame_queue.get()
            cv2.imshow("Right Camera", right_frame)

        # Compute 3D position if ball detected in both cameras
        if not left_pos_queue.empty() and not right_pos_queue.empty():
            x1 = left_pos_queue.get()
            x2 = right_pos_queue.get()
            
            # Calculate 3D position
            X = calculate_3d_position(x1, x2, R, T, pose_rotation, pose_translation)
            print(f"3D Position: {X.flatten()}")
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Clean up resources
    cleanup_resources(left_camera, right_camera, left_tracker, right_tracker, stop_event)
""" 