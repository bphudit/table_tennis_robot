import threading
from pypylon import pylon
import cv2
import numpy as np
import time
from queue import Queue

class BallDetectionCamera(threading.Thread):
    def __init__(self, camera_index, intrinsic_inv, lower_limit, upper_limit):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.current_frame = None
        self.current_mask = None
        self.camera = None
        self.converter = pylon.ImageFormatConverter()
        self.x_queue = Queue(maxsize=1)
        
        # Ball detection parameters
        self.intrinsic_inv = intrinsic_inv
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        
        # Set up format converter
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
    def initialize_camera(self):
        try:
            # Get all available cameras
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            if not devices:
                raise Exception("No camera found!")
            
            if self.camera_index >= len(devices):
                raise Exception(f"Camera index {self.camera_index} is out of range!")
            
            # Create and open camera
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[self.camera_index]))
            self.camera.Open()
            
            # Set camera parameters
            self.camera.ExposureTime.SetValue(10000)
            self.camera.Gain.SetValue(0)
            
            # Start grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print(f"Camera {self.camera_index} initialized successfully")
            
        except Exception as e:
            print(f"Error initializing camera {self.camera_index}: {str(e)}")
            raise

    def process_frame(self, color_image):
        u, v = None, None
        
        # Preprocessing for ball detection
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        blurred_image = cv2.GaussianBlur(hsv_image, (7, 7), 0)
        mask = cv2.inRange(blurred_image, self.lower_limit, self.upper_limit)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        self.current_mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # if 10 <= cv2.contourArea(contour) <= 300:
            x, y, w, h = cv2.boundingRect(contour)
            u, v = x + w // 2, y + h // 2
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # break
        
        if u is not None and v is not None:
            if self.x_queue.full():
                self.x_queue.get()
            x = self.intrinsic_inv @ np.array([u, v, 1])
            self.x_queue.put(x)
        
        return color_image
    
    def run(self):
        self.running = True
        self.initialize_camera()
        
        while self.running:
            if self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    # Convert to BGR format using Pylon converter
                    image = self.converter.Convert(grab_result)
                    # Convert to numpy array
                    frame = image.GetArray()
                    # Process frame for ball detection
                    self.current_frame = self.process_frame(frame)
                
                grab_result.Release()
    
    def stop(self):
        self.running = False
        if self.camera:
            self.camera.Close()
    
    def get_frame(self):
        return self.current_frame
    
    def get_mask(self):
        return self.current_mask
    
    def get_ball_position(self):
        if not self.x_queue.empty():
            return self.x_queue.get()
        return None

def main():
    # Example usage with dummy parameters
    K1_inv = np.eye(3)  # Replace with actual inverse intrinsic matrix
    lower_limit = np.array([0, 127, 140])  # Replace with actual values
    upper_limit = np.array([87, 255, 255])
    
    try:
        # Create camera threads
        camera1 = BallDetectionCamera(0, K1_inv, lower_limit, upper_limit)
        camera2 = BallDetectionCamera(1, K1_inv, lower_limit, upper_limit)
        
        # Start camera threads
        camera1.start()
        camera2.start()
        
        # Wait for cameras to initialize
        time.sleep(2)
        
        while True:
            # Get frames from both cameras
            frame1 = camera1.get_frame()
            frame2 = camera2.get_frame()
            
            # Get masks from both cameras
            mask1 = camera1.get_mask()
            mask2 = camera2.get_mask()
            
            # Get ball positions
            pos1 = camera1.get_ball_position()
            pos2 = camera2.get_ball_position()
            
            if pos1 is not None:
                print(f"Camera 1 ball position: {pos1}")
            if pos2 is not None:
                print(f"Camera 2 ball position: {pos2}")
            
            if frame1 is not None and frame2 is not None:
                # Display original frames
                cv2.imshow('Camera 1', frame1)
                cv2.imshow('Camera 2', frame2)
                
                # Display mask frames
                if mask1 is not None:
                    cv2.imshow('Camera 1 Mask', mask1)
                if mask2 is not None:
                    cv2.imshow('Camera 2 Mask', mask2)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except KeyboardInterrupt:
        print("Stopping cameras...")
    finally:
        camera1.stop()
        camera2.stop()
        camera1.join()
        camera2.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
