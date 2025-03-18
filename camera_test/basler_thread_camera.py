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
            self.camera.ExposureTime.SetValue(9000)  # Set exposure time to 6000
            self.camera.Gain.SetValue(0)  # Set gain to 0
            
            # Start grabbing
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
                    # Convert to BGR format using Pylon converter
                    image = self.converter.Convert(grab_result)
                    # Convert to numpy array
                    self.current_frame = image.GetArray()
                
                grab_result.Release()
    
    def stop(self):
        self.running = False
        if self.camera:
            self.camera.Close()
    
    def get_frame(self):
        return self.current_frame

def main():
    try:
        # Create camera threads
        camera1 = BaslerCamera(0)  # First camera
        camera2 = BaslerCamera(1)  # Second camera
        
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
                # Display the frames
                cv2.imshow('Camera 1', frame1)
                cv2.imshow('Camera 2', frame2)
                
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except KeyboardInterrupt:
        print("Stopping cameras...")
    finally:
        # Clean up
        camera1.stop()
        camera2.stop()
        camera1.join()
        camera2.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 