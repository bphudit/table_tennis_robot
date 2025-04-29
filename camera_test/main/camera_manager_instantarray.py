import cv2
import numpy as np
from pypylon import pylon
import time
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
            cam.MaxNumBuffer = 10
        # Prepare converter for OpenCV
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        # Start grabbing
        if mode == 0:
            self.cams.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        elif mode == 1:
            self.cams.StartGrabbing(pylon.GrabStrategy_OneByOne)
        print("Grabbing started. Awaiting hardware triggers...")
        
    def get_synchronized_frames(self, timeout_ms=5000):
        if not self.cams.IsGrabbing():
            return None, None, None, None

        frames = {}
        timestamps = {}

        while True:
            # Retrieve next available frame from any camera
            grab = self.cams.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            if not grab.GrabSucceeded():
                grab.Release()
                continue

            ctx = grab.GetCameraContext()  # 0 or 1
            bid = grab.BlockID
            tms = grab.TimeStamp
            img = self.converter.Convert(grab).GetArray()
            grab.Release()

            # Store by block ID and camera context
            frames.setdefault(bid, {})[ctx] = img
            # Store timestamp for this block ID
            timestamps.setdefault(bid, {})[ctx] = tms
            # print(timestamps)
            # print(bid, timestamps[bid])
            # Check if we have both cameras for this block ID
            if bid in frames and len(frames[bid]) == 2:
                img0 = frames[bid][0]
                img1 = frames[bid][1]
                return img0, img1, bid, timestamps[bid][0]

        # If we reach here, failed to get synchronized pair
        return None, None, None, None

    def close(self):
        """Stop grabbing and clean up resources."""
        self.cams.StopGrabbing()
        for cam in self.cams:
            if cam.IsOpen():
                cam.Close()

