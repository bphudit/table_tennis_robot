import cv2
import numpy as np
from pypylon import pylon

class CameraManager:
    def __init__(self, exposure_time=6000, gain=10, mode=0):
        """Initialize the camera manager with two synchronized Basler cameras.
        
        Args:
            exposure_time: Camera exposure time in microseconds (default: 6000)
        """
        # Discover cameras
        self.tl_factory = pylon.TlFactory.GetInstance()
        self.devices = self.tl_factory.EnumerateDevices()

        if len(self.devices) < 2:
            raise RuntimeError("At least two Basler cameras are required")
        
        # Initialize camera array
        self.cams = pylon.InstantCameraArray(2)
        for i, cam in enumerate(self.cams):
            cam.Attach(self.tl_factory.CreateDevice(self.devices[i]))
            cam.Open()
            
            # Configure hardware trigger on Line1
            cam.TriggerSelector.SetValue("FrameStart")
            cam.TriggerSource.SetValue("Line1")
            cam.TriggerActivation.SetValue("RisingEdge")
            cam.TriggerMode.SetValue("On")  # enable external trigger mode
            cam.ExposureTime.SetValue(exposure_time)
            cam.Gain.SetValue(gain)
        
        # Prepare converter for OpenCV
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        # Start grabbing with latest-only strategy
        if mode == 0:
            self.cams.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        elif mode == 1:
            self.cams.StartGrabbing(pylon.GrabStrategy_OneByOne)
        print("Grabbing started. Awaiting hardware triggers...")
        
    def get_synchronized_frames(self, timeout_ms=5000):
        """Get synchronized frames from both cameras.
        
        Args:
            timeout_ms: Timeout in milliseconds for frame retrieval
            
        Returns:
            tuple: (img1, img2) as numpy arrays if synchronized frames are available,
                  (None, None) if frames couldn't be grabbed or weren't synchronized
        """
        if not self.cams.IsGrabbing():
            return None, None
            
        try:
            result1 = self.cams[0].RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            result2 = self.cams[1].RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            
            if result1.GrabSucceeded() and result2.GrabSucceeded():
                img1 = self.converter.Convert(result1).GetArray()
                img2 = self.converter.Convert(result2).GetArray()
                
                # Check if frames are synchronized (same BlockID)
                if result1.BlockID == result2.BlockID:
                    return img1, img2, result1.BlockID, result2.BlockID
                else:
                    return None, None, result1.BlockID, result2.BlockID
            
        except pylon.TimeoutException:
            print("Timeout occurred while waiting for images")
        except Exception as e:
            print(f"Error retrieving frames: {e}")
        finally:
            if result1 is not None:
                result1.Release()
            if result2 is not None:
                result2.Release()
            
        
    
    def close(self):
        """Stop grabbing and clean up resources."""
        self.cams.StopGrabbing()
        for cam in self.cams:
            if cam.IsOpen():
                cam.Close() 