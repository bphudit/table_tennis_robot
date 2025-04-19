import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class BallDetector:
    def __init__(self, max_workers=2, lower_limit=None, upper_limit=None):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Default HSV values for orange ping pong ball
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        
    def set_hsv_limits(self, lower_limit, upper_limit):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        
    def detect_ball(self, frame, lower_limit=None, upper_limit=None):
        """
        Detect a ball in a single frame using HSV color thresholding
        Returns frame with detection rectangle drawn
        """
        if lower_limit is None:
            lower_limit = self.lower_limit
        if upper_limit is None:
            upper_limit = self.upper_limit
            
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred_image = cv2.GaussianBlur(hsv_image, (7, 7), 0)
        mask = cv2.inRange(blurred_image, lower_limit, upper_limit)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
        return frame
        
    def process_stereo_frames(self, frame1, frame2, lower_limit=None, upper_limit=None):
        """
        Process two frames in parallel using thread pool
        """
        if lower_limit is None:
            lower_limit = self.lower_limit
        if upper_limit is None:
            upper_limit = self.upper_limit
            
        future1 = self.thread_pool.submit(self.detect_ball, frame1, lower_limit, upper_limit)
        future2 = self.thread_pool.submit(self.detect_ball, frame2, lower_limit, upper_limit)

        frame1_processed = future1.result()
        frame2_processed = future2.result()
        
        return frame1_processed, frame2_processed
    
    def close(self):
        """
        Clean up resources
        """
        self.thread_pool.shutdown() 