import cv2
import numpy as np
import os
from scipy.io import loadmat
from ball_triangulation_cv import BallTriangulatorP

class TableCalibrationCV:
    def __init__(self):
        # Load stereo camera parameters
        data = loadmat('src/stereoParameters.mat')
        stereo_params = {
            'K1': data['K1'],
            'K2': data['K2'],
            'R': data['R'],
            'T': data['T'].reshape((3, 1))
        }
        
        # Create an empty pose_params structure (will be calculated)
        self.pose_params = {
            'rotation_matrix': None,
            'translation_matrix': None
        }
        
        # Initialize the triangulator with current parameters
        # It will be updated once we have the actual pose parameters
        self.triangulator = BallTriangulatorP(stereo_params, self.pose_params)
        
        # Initialize points storage for both cameras
        self.reset_points()
        self.current_window = None
        self.scale_factor = 2.0  # 2x larger display
        
    def reset_points(self):
        """Reset all points for both cameras."""
        self.points = {'left': [], 'right': []}
        self.triangulated_points_3d = []
        
    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_window and len(self.points[param]) < 4:
            camera = param
            # Convert clicked coordinates back to original scale
            original_x = int(x / self.scale_factor)
            original_y = int(y / self.scale_factor)
            
            # Append the point
            self.points[camera].append((original_x, original_y))
            
            # Draw point and coordinates on image
            cv2.circle(self.img, (x, y), 4, (0, 0, 255), -1)
            text = f"{len(self.points[camera])}. ({original_x}, {original_y})"
            cv2.putText(self.img, text, (x, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow(self.current_window, self.img)
            cv2.waitKey(1)  # Force window refresh
            
            print(f"Marked point {len(self.points[camera])} for {camera} camera")
            
            # If we have all 4 corners, enable completion
            if len(self.points[camera]) == 4:
                print(f"\nAll 4 corners marked for {camera} camera. Press 'c' to continue or 'r' to reset.")
            
    def capture_corners(self, image_path, camera):
        """Capture the 4 corners of the table tennis table for specified camera."""
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Error: Could not read image from {image_path}")
            return False
            
        # Resize image to make it larger
        height, width = original_img.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        self.img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        # Create a fresh window
        if self.current_window:
            cv2.destroyWindow(self.current_window)
        
        self.current_window = f"Image - {camera.capitalize()} Camera"
        cv2.namedWindow(self.current_window)
        cv2.imshow(self.current_window, self.img)
        cv2.setMouseCallback(self.current_window, self.click_event, camera)
        
        print(f"\nPlease mark 4 corners for {camera} camera in order:")
        print("1. Top-left corner")
        print("2. Top-right corner")
        print("3. Bottom-right corner")
        print("4. Bottom-left corner")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # Reset points for current camera
                self.points[camera] = []
                self.img = cv2.resize(original_img.copy(), (new_width, new_height), 
                                    interpolation=cv2.INTER_AREA)
                cv2.imshow(self.current_window, self.img)
                print(f"\nPoints reset for {camera} camera. Please mark the 4 corners again.")
            elif key == ord('c') and len(self.points[camera]) == 4:  # Continue to next step
                cv2.destroyWindow(self.current_window)
                self.current_window = None
                return True
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                self.current_window = None
                return False

    def triangulate_corners(self):
        """Triangulate the 3D coordinates of corners using OpenCV's triangulation."""
        # Create output directories if they don't exist
        os.makedirs("src/table_pose", exist_ok=True)
        
        # Check if we have both sets of corner points
        if len(self.points['left']) != 4 or len(self.points['right']) != 4:
            print("Error: Need exactly 4 corner points for each camera")
            return None, None
        
        # Triangulate each corner point
        for i in range(4):
            left_point = self.points['left'][i]
            right_point = self.points['right'][i]
            
            # Use OpenCV triangulation
            # We need to pass points as arrays with shape (2,1)
            pts1 = np.array([[left_point[0]], [left_point[1]]], dtype=np.float64)
            pts2 = np.array([[right_point[0]], [right_point[1]]], dtype=np.float64)
            
            # Use projection matrices from triangulator
            homogeneous_4d = cv2.triangulatePoints(
                self.triangulator.P1, 
                self.triangulator.P2, 
                pts1, pts2
            )
            
            # Convert from homogeneous to 3D (divide by last row)
            point_3d = homogeneous_4d[:3, 0] / homogeneous_4d[3, 0]
            self.triangulated_points_3d.append(point_3d.reshape((3, 1)))
        
        # Save the raw triangulated points for later reference
        np.savez("src/table_pose/triangulated_corners.npz", corners=self.triangulated_points_3d)
        print("Raw triangulated corners saved to 'src/table_pose/triangulated_corners.npz'")
        
        return self.triangulated_points_3d

    def calculate_transformation(self):
        """Calculate the transformation matrix using triangulated corner points."""
        # First, triangulate the corners if not already done
        if not self.triangulated_points_3d:
            points_3d = self.triangulate_corners()
            if not points_3d:
                return None, None
        
        # Extract the triangulated corner points
        result = self.triangulated_points_3d
        
        # Calculate basis vectors and ensure orthogonality using Gram-Schmidt process
        # Use standard table corner convention: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
        vx = result[1] - result[0]  # Vector from point 0 to point 1 (width)
        vy = result[3] - result[0]  # Vector from point 0 to point 3 (length)
        
        # Normalize x-axis
        ux = vx / np.linalg.norm(vx)
        
        # Make y-axis orthogonal to x-axis using Gram-Schmidt
        vy_proj = np.dot(vy.flatten(), ux.flatten()) * ux  # Project vy onto ux
        vy_orth = vy - vy_proj  # Make vy orthogonal to ux
        uy = vy_orth / np.linalg.norm(vy_orth)  # Normalize
        
        # Calculate z-axis as cross product (ensures right-hand coordinate system)
        uz = np.cross(ux.flatten(), uy.flatten())
        uz = uz.reshape((3, 1)) / np.linalg.norm(uz)  # Normalize

        # Construct the rotation matrix
        R = np.hstack((ux, uy, uz))
        T = result[0]  # Translation = position of top-left corner

        # Save transformation parameters
        np.savez("src/pose_parameters.npz", rotation_matrix=R, translation_matrix=T)
        
        # Update the pose parameters in the instance
        self.pose_params['rotation_matrix'] = R
        self.pose_params['translation_matrix'] = T
        
        # Print verification and transformed points
        print("\nTransformation calculated successfully!")
        print("Orthogonality check:", np.allclose(R @ R.T, np.eye(3), atol=1e-6))
        print("Determinant (should be close to 1):", np.linalg.det(R))
        
        print("\nTransformed corner points (in table coordinates, centimeters):")
        table_dimensions = {
            'width': 152.5,  # Standard table width in cm
            'length': 274    # Standard table length in cm
        }
        
        # Expected corner coordinates in table coordinates
        expected_corners = [
            [0, 0, 0],                    # Top-left
            [table_dimensions['width'], 0, 0],  # Top-right
            [table_dimensions['width'], table_dimensions['length'], 0],  # Bottom-right
            [0, table_dimensions['length'], 0]   # Bottom-left
        ]
        
        for i, point in enumerate(result):
            point_cam = point
            # Transform to table coordinates
            transformed_point = R.T @ (point_cam - T)
            # Scale to centimeters
            transformed_point_cm = transformed_point / 10
            
            print(f"Corner {i+1}: {transformed_point_cm.flatten()} cm (Expected: {expected_corners[i]})")
        
        return R, T

def main():
    calibrator = TableCalibrationCV()
    print("We'll calibrate the table using OpenCV triangulation method.")
    print("Please mark the 4 corners of the table tennis table in this order:")
    print("1. Top-left corner")
    print("2. Top-right corner")
    print("3. Bottom-right corner")
    print("4. Bottom-left corner")
    print("\nPress 'r' to reset points if you make a mistake")
    print("Press 'c' to continue after marking all 4 points")
    print("Press 'q' to quit\n")
    
    # Capture corners for left camera
    print("\nStarting with LEFT camera...")
    if not calibrator.capture_corners("camera_test/left/img0.png", 'left'):
        print("\nCalibration cancelled.")
        return
        
    # Capture corners for right camera
    print("\nNow for RIGHT camera...")
    if not calibrator.capture_corners("camera_test/right/img0.png", 'right'):
        print("\nCalibration cancelled.")
        return
    
    # Triangulate corners and calculate transformation
    R, T = calibrator.calculate_transformation()
    if R is not None and T is not None:
        print("\nCalibration complete! Transformation matrices saved to 'pose_parameters.npz'")
    else:
        print("\nCalibration failed.")

if __name__ == "__main__":
    main() 