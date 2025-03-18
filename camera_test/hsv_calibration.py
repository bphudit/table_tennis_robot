import threading
from pypylon import pylon
import cv2
import numpy as np
import time

class BaslerCamera(threading.Thread):
    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.current_frame = None
        self.camera = None
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
            
            self.camera.ExposureTime.SetValue(10000)
            self.camera.Gain.SetValue(0)
            
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

def nothing(x):
    pass

def main():
    try:
        # Create windows for trackbars
        cv2.namedWindow('HSV Trackbars')
        cv2.resizeWindow('HSV Trackbars', 1200, 1200)
        
        # Create trackbars
        cv2.createTrackbar('H min', 'HSV Trackbars', 0, 179, nothing)
        cv2.createTrackbar('H max', 'HSV Trackbars', 81, 179, nothing)
        cv2.createTrackbar('S min', 'HSV Trackbars', 109, 255, nothing)
        cv2.createTrackbar('S max', 'HSV Trackbars', 255, 255, nothing)
        cv2.createTrackbar('V min', 'HSV Trackbars', 141, 255, nothing)
        cv2.createTrackbar('V max', 'HSV Trackbars', 255, 255, nothing)
        
        # Create camera threads
        camera1 = BaslerCamera(0)
        camera2 = BaslerCamera(1)
        
        # Start camera threads
        camera1.start()
        camera2.start()
        
        # Wait for cameras to initialize
        time.sleep(2)
        
        while True:
            # Get frames from both cameras
            frame1 = camera1.get_frame()
            frame2 = camera2.get_frame()
            
            if frame1 is not None and frame2 is not None:
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
                
                # Apply mask and preprocessing
                mask1 = cv2.inRange(hsv1, lower, upper)
                mask2 = cv2.inRange(hsv2, lower, upper)
                
                # Apply morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
                mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
                
                # Find and draw contours
                contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                frame1_with_contours = frame1.copy()
                frame2_with_contours = frame2.copy()
                
                # Draw contours that match ball size criteria
                for contour in contours1:
                    if 10 <= cv2.contourArea(contour) <= 300:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame1_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                for contour in contours2:
                    if 10 <= cv2.contourArea(contour) <= 300:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame2_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display results
                cv2.imshow('Camera 1 Original', frame1_with_contours)
                cv2.imshow('Camera 1 Mask', mask1)
                cv2.imshow('Camera 2 Original', frame2_with_contours)
                cv2.imshow('Camera 2 Mask', mask2)
                
                # Print current HSV values
                print(f"\rCurrent HSV values - Lower: [{h_min}, {s_min}, {v_min}], Upper: [{h_max}, {s_max}, {v_max}]", end='')
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except KeyboardInterrupt:
        print("\nStopping cameras...")
    finally:
        camera1.stop()
        camera2.stop()
        camera1.join()
        camera2.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
