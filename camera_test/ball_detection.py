import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class BallDetector:
    def __init__(self, max_workers=2, lower_limit=None, upper_limit=None, downsample_ratio=1.0):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Default HSV values for orange ping pong ball
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.downsample_ratio = downsample_ratio
        
    def set_hsv_limits(self, lower_limit, upper_limit):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        
    def detect_ball(self, frame, lower_limit=None, upper_limit=None):
        """
        Detect a ball in a single frame using HSV color thresholding
        Returns:
            - processed_frame: frame with detection rectangle drawn
            - ball_coords: list of (u, v) coordinates of detected balls (centers of rectangles)
        """
        if lower_limit is None:
            lower_limit = self.lower_limit
        if upper_limit is None:
            upper_limit = self.upper_limit
            
        # Downsample the frame if ratio is less than 1.0
        original_frame = frame.copy()
        if self.downsample_ratio < 1.0:
            height, width = frame.shape[:2]
            new_height = int(height * self.downsample_ratio)
            new_width = int(width * self.downsample_ratio)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred_image = cv2.GaussianBlur(hsv_image, (7, 7), 0)
        mask = cv2.inRange(blurred_image, lower_limit, upper_limit)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Scale for contour area threshold based on downsample ratio
        min_area = 10000 * (self.downsample_ratio ** 2)
        
        detected_rectangles = []
        ball_coords = None
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                if self.downsample_ratio < 1.0:
                    # Scale back to original frame dimensions
                    scale = 1.0 / self.downsample_ratio
                    x, y = int(x * scale), int(y * scale)
                    w, h = int(w * scale), int(h * scale)
                    cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                detected_rectangles.append((x, y, w, h))
                
                # Calculate center point (u, v)
                u, v = x + w // 2, y + h // 2
                ball_coords = (u, v)

                break
                
        processed_frame = original_frame if self.downsample_ratio < 1.0 else frame
        return processed_frame, ball_coords
        
    def process_stereo_frames(self, frame1, frame2, lower_limit=None, upper_limit=None):
        """
        Process two frames in parallel using thread pool
        Returns:
            - processed frames with detections
            - coordinates of balls in each frame
        """
        if lower_limit is None:
            lower_limit = self.lower_limit
        if upper_limit is None:
            upper_limit = self.upper_limit
            
        future1 = self.thread_pool.submit(self.detect_ball, frame1, lower_limit, upper_limit)
        future2 = self.thread_pool.submit(self.detect_ball, frame2, lower_limit, upper_limit)

        frame1_processed, coords1 = future1.result()
        frame2_processed, coords2 = future2.result()
        
        return (frame1_processed, coords1), (frame2_processed, coords2)
    
    def close(self):
        """
        Clean up resources
        """
        self.thread_pool.shutdown() 