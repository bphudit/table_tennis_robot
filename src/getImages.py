import cv2
import numpy as np
import time
from pypylon import pylon
import threading

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
            self.camera.Gain.SetValue(15)
            
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

def main():
    try:
        # Create camera threads
        camera_left = BaslerCamera(0)
        camera_right = BaslerCamera(1)
        
        # Start camera threads
        camera_left.start()
        camera_right.start()
        
        # Wait for cameras to initialize
        time.sleep(2)
        
        num = 0
        
        while True:
            # Get frames from both cameras
            frame_left = camera_left.get_frame()
            frame_right = camera_right.get_frame()
            
            if frame_left is not None and frame_right is not None:
                # Get the current frame size
                height, width, _ = frame_left.shape

                # Resize the frame
                scale_percent = 40
                new_width = int(width * scale_percent / 100)
                new_height = int(height * scale_percent / 100)
                dim = (new_width, new_height)
                resized_left = cv2.resize(frame_left, dim, interpolation=cv2.INTER_AREA)
                resized_right = cv2.resize(frame_right, dim, interpolation=cv2.INTER_AREA)

                cv2.imshow('Left Camera', resized_left)
                cv2.imshow('Right Camera', resized_right)

                k = cv2.waitKey(5)

                if k == 27:  # ESC key
                    break
                elif k == ord('s'):
                    cv2.imwrite('camera_test/right/img' + str(num) + '.png', frame_right)
                    cv2.imwrite('camera_test/left/img' + str(num) + '.png', frame_left)
                    print("image saved!")
                    num += 1
                elif k == ord('q'):
                    break
        
    except KeyboardInterrupt:
        print("\nStopping cameras...")
    finally:
        camera_left.stop()
        camera_right.stop()
        camera_left.join()
        camera_right.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()