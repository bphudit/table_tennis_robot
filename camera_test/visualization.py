import cv2
import time

class FrameVisualizer:
    def __init__(self, scale_factor=0.5):
        self.scale_factor = scale_factor
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
    def resize_frame(self, frame):
        """Resize a frame by the scale factor"""
        return cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        
    def update_fps(self):
        """Update the FPS calculation"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps
        
    def add_fps_info(self, frame, fps, processing_time_ms):
        """Add FPS and processing time text to a frame"""
        fps_text = f"FPS: {fps:.1f}"
        process_time_text = f"Process time: {processing_time_ms:.1f}ms"
        
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, process_time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
        
    def display_frames(self, frame1, frame2, process_time_ms=0):
        """Display two frames with FPS and processing time info"""
        frame1, coords1 = frame1
        frame2, coords2 = frame2

        # Update FPS
        current_fps = self.update_fps()
        if coords1 is not None:
            cv2.circle(frame1, coords1, 5, (0, 0, 255), -1)
        if coords2 is not None:
            cv2.circle(frame2, coords2, 5, (0, 0, 255), -1)
        
        # Resize frames
        frame1_resized = self.resize_frame(frame1)
        frame2_resized = self.resize_frame(frame2)
        
        # Add FPS and processing time info
        frame1_resized = self.add_fps_info(frame1_resized, current_fps, process_time_ms)
        frame2_resized = self.add_fps_info(frame2_resized, current_fps, process_time_ms)
        
        # Display frames
        cv2.imshow("Camera 1", frame1_resized)
        cv2.imshow("Camera 2", frame2_resized)
        
        # Print FPS to console occasionally
        if self.frame_count % 10 == 0:
            print(f"Current FPS: {current_fps:.1f}, Processing time per frame: {process_time_ms:.1f}ms")
            
        # Return key press (for checking if 'q' was pressed)
        return cv2.waitKey(1) & 0xFF
        
    def close(self):
        """Close all OpenCV windows"""
        cv2.destroyAllWindows() 