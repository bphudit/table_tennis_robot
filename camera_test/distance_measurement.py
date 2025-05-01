#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
from scipy.io import loadmat
import math
import sys
from main.ball_triangulation_cv import BallTriangulatorP


class DistanceMeasurer:
    def __init__(self, stereo_params, pose_params, left_images_dir='left_camera', right_images_dir='right_camera'):
        # Initialize triangulator using camera parameters
        self.triangulator = BallTriangulatorP(stereo_params, pose_params, coord_frame = 'camera')
        
        # Initialize directories for left and right camera images
        self.left_images_dir = left_images_dir
        self.right_images_dir = right_images_dir
        
        # Store clicked points
        self.left_points = []
        self.right_points = []
        self.points_3d = []
        
        # Image pairs
        self.image_pairs = []
        self.current_pair_index = 0
        
        # Current images
        self.left_img = None
        self.right_img = None
        
        # Flags for controlling UI
        self.selecting_left = True  # Start by selecting in left image
        self.point_index = 0        # Track which point (first or second) we're selecting
        
        # Initialize the display
        self.window_name = "Stereo Distance Measurement"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
    def load_image_pairs(self):
        """Load matching image pairs from left and right directories"""
        # First check camera_test/left and camera_test/right
        left_files = sorted(glob.glob(os.path.join('camera_test', 'left', '*.png')))
        right_files = sorted(glob.glob(os.path.join('camera_test', 'right', '*.png')))
        
        # If those directories don't exist or are empty, try the alternate directories
        if not left_files or not right_files:
            left_files = sorted(glob.glob(os.path.join(self.left_images_dir, '*.png')))
            right_files = sorted(glob.glob(os.path.join(self.right_images_dir, '*.png')))
        
        # Pair up images with the same names
        left_basenames = [os.path.basename(f) for f in left_files]
        right_basenames = [os.path.basename(f) for f in right_files]
        
        # Find common files
        common_basenames = set(left_basenames).intersection(set(right_basenames))
        
        for basename in sorted(common_basenames):
            left_path = next(f for f in left_files if os.path.basename(f) == basename)
            right_path = next(f for f in right_files if os.path.basename(f) == basename)
            self.image_pairs.append((left_path, right_path))
        
        print(f"Loaded {len(self.image_pairs)} image pairs")
        
        if not self.image_pairs:
            raise ValueError("No matching image pairs found in specified directories")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Determine which half of the screen was clicked
            height, width = self.displayed_img.shape[:2]
            
            if x < width // 2:  # Left image
                # Only allow clicks on left image when we're in left selection mode
                if self.selecting_left:
                    # Calculate coordinates in original left image
                    orig_x = int(x * (self.left_img.shape[1] / (width // 2)))
                    orig_y = int(y * (self.left_img.shape[0] / height))
                    
                    if self.point_index < 2:
                        if len(self.left_points) <= self.point_index:
                            self.left_points.append((orig_x, orig_y))
                        else:
                            self.left_points[self.point_index] = (orig_x, orig_y)
                    
                    print(f"Selected point {self.point_index + 1} in left image: ({orig_x}, {orig_y})")
                    
                    # Switch to right image selection
                    self.selecting_left = False
                    self.update_display()
            else:  # Right image
                # Only allow clicks on right image when we're in right selection mode
                if not self.selecting_left:
                    # Calculate coordinates in original right image
                    orig_x = int((x - width // 2) * (self.right_img.shape[1] / (width // 2)))
                    orig_y = int(y * (self.right_img.shape[0] / height))
                    
                    if self.point_index < 2:
                        if len(self.right_points) <= self.point_index:
                            self.right_points.append((orig_x, orig_y))
                        else:
                            self.right_points[self.point_index] = (orig_x, orig_y)
                    
                    print(f"Selected point {self.point_index + 1} in right image: ({orig_x}, {orig_y})")
                    
                    # After selecting a point in both images, triangulate to get 3D position
                    if len(self.left_points) > self.point_index and len(self.right_points) > self.point_index:
                        self.triangulate_point(self.point_index)
                        
                        # Move to next point
                        self.point_index += 1
                        
                        # If we've selected two points, calculate distance
                        if self.point_index >= 2:
                            self.calculate_distance()
                            
                            # Reset for the next pair of points
                            self.point_index = 0
                            
                    # Switch back to left image for next point
                    self.selecting_left = True
                    self.update_display()
    
    def triangulate_point(self, index):
        """Triangulate 3D position from corresponding image points"""
        left_point = self.left_points[index]
        right_point = self.right_points[index]
        
        # Use the triangulator to calculate 3D position
        point_3d = self.triangulator.process_frames_thread(
            np.array([left_point[0], left_point[1]]), 
            np.array([right_point[0], right_point[1]]),
            0
        )
        
        if point_3d is not None:
            if len(self.points_3d) <= index:
                self.points_3d.append(point_3d)
            else:
                self.points_3d[index] = point_3d
                
            print(f"3D position of point {index + 1}: [{point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f}]")
        else:
            print(f"Failed to triangulate point {index + 1}. Please try selecting the point again.")
    
    def calculate_distance(self):
        """Calculate the Euclidean distance between two 3D points"""
        if len(self.points_3d) >= 2:
            p1 = self.points_3d[0]
            p2 = self.points_3d[1]
            
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((p1 - p2) ** 2))
            
            print(f"\n--- DISTANCE MEASUREMENT ---")
            print(f"Point 1: [{p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f}] mm")
            print(f"Point 2: [{p2[0]:.2f}, {p2[1]:.2f}, {p2[2]:.2f}] mm")
            print(f"Distance: {distance:.2f} mm")
            print(f"----------------------------\n")
            
            # Display the distance on the image
            self.update_display(f"Distance: {distance:.2f} mm")
    
    def update_display(self, text=None):
        """Update the display with current images and points"""
        if self.left_img is None or self.right_img is None:
            return
        
        # Create copies of the images for drawing
        left_display = self.left_img.copy()
        right_display = self.right_img.copy()
        
        # Draw selected points
        for i, point in enumerate(self.left_points):
            cv2.circle(left_display, point, 5, (0, 255, 0), -1)
            cv2.putText(left_display, f"P{i+1}", (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        for i, point in enumerate(self.right_points):
            cv2.circle(right_display, point, 5, (0, 255, 0), -1)
            cv2.putText(right_display, f"P{i+1}", (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Highlight the image where the next click should occur
        if self.selecting_left:
            cv2.putText(left_display, "Click here", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(right_display, "Click here", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add prompt text
        if self.point_index == 0:
            prompt = "Select point 1"
        else:
            prompt = "Select point 2"
            
        # Add the distance text if provided
        if text:
            prompt = text
        
        # Combine images horizontally
        self.displayed_img = np.hstack((left_display, right_display))
        
        # Add the prompt text
        cv2.putText(self.displayed_img, prompt, (10, self.displayed_img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the filename
        if self.current_pair_index < len(self.image_pairs):
            left_path, right_path = self.image_pairs[self.current_pair_index]
            filename = f"Image: {os.path.basename(left_path)}"
            cv2.putText(self.displayed_img, filename, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the combined image
        cv2.imshow(self.window_name, self.displayed_img)
    
    def load_current_pair(self):
        """Load the current image pair"""
        if self.current_pair_index < len(self.image_pairs):
            left_path, right_path = self.image_pairs[self.current_pair_index]
            
            self.left_img = cv2.imread(left_path)
            self.right_img = cv2.imread(right_path)
            
            # Reset points
            self.left_points = []
            self.right_points = []
            self.points_3d = []
            self.point_index = 0
            self.selecting_left = True
            
            self.update_display()
            print(f"Loaded image pair {self.current_pair_index + 1}/{len(self.image_pairs)}")
    
    def next_pair(self):
        """Move to the next image pair"""
        if self.current_pair_index < len(self.image_pairs) - 1:
            self.current_pair_index += 1
            self.load_current_pair()
    
    def prev_pair(self):
        """Move to the previous image pair"""
        if self.current_pair_index > 0:
            self.current_pair_index -= 1
            self.load_current_pair()
    
    def run(self):
        """Run the distance measurement application"""
        try:
            # Load image pairs
            self.load_image_pairs()
            
            if not self.image_pairs:
                print("No image pairs found. Exiting.")
                return
            
            # Load the first pair
            self.load_current_pair()
            
            # Set mouse callback
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
            print("Instructions:")
            print("1. Click on a point in the left image")
            print("2. Click on the corresponding point in the right image")
            print("3. Repeat for a second point")
            print("4. The 3D distance will be calculated")
            print("\nControls:")
            print("  'n' - next image pair")
            print("  'p' - previous image pair")
            print("  'r' - reset current points")
            print("  'q' - quit")
            
            # Main loop
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.next_pair()
                elif key == ord('p'):
                    self.prev_pair()
                elif key == ord('r'):
                    # Reset points
                    self.left_points = []
                    self.right_points = []
                    self.points_3d = []
                    self.point_index = 0
                    self.selecting_left = True
                    self.update_display()
        
        finally:
            cv2.destroyAllWindows()


def main():
    # Load stereo parameters
    stereo_params = loadmat('src/stereoParameters.mat')
    pose_params = np.load('src/pose_parameters.npz')
    
    # Initialize and run the distance measurer
    measurer = DistanceMeasurer(stereo_params, pose_params)
    measurer.run()


if __name__ == "__main__":
    main() 