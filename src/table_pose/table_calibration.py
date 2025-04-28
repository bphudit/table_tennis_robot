import cv2
import numpy as np
from scipy.io import loadmat
import os
import sys

# Add the root directory to the Python path to enable importing from camera_test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from camera_test.main.ball_triangulation_cv import BallTriangulatorP

class TableCalibration:
    def __init__(self): 
        # Load stereo camera parameters
        data = loadmat('src/stereoParameters.mat')
        self.K1 = data['K1']
        self.K2 = data['K2']
        self.K1_inv = np.linalg.inv(self.K1)
        self.K2_inv = np.linalg.inv(self.K2)
        self.R_cam = np.linalg.inv(data['R'])
        self.T_cam = -self.R_cam @ data['T'].reshape((3, 1))
        
        # Initialize points storage for both cameras
        self.reset_points()
        self.current_window = None
        self.scale_factor = 2.0  # 2x larger display

        self.triangulator = BallTriangulatorP(stereo_params=data, pose_params=None, coord_frame = 'camera')
        
    def reset_points(self):
        """Reset all points for both cameras."""
        self.points = {'left': [], 'right': []}
        self.pixel2points = {'left': [], 'right': []}
        
    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_window and len(self.points[param]) < 4:
            camera = param
            # Convert clicked coordinates back to original scale
            original_x = int(x / self.scale_factor)
            original_y = int(y / self.scale_factor)
            
            # Append the point and convert to camera coordinates
            self.points[camera].append((original_x, original_y))
            
            # Use appropriate camera matrix for conversion
            if camera == 'left':
                self.pixel2points[camera].append(self.K1_inv @ np.array([original_x, original_y, 1]))
                
            else:
                self.pixel2points[camera].append(self.K2_inv @ np.array([original_x, original_y, 1]))
            
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
                self.pixel2points[camera] = []
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

    def calculate_transformation(self):
        """Calculate the transformation matrix using points from both cameras."""
        # Create table_pose directory if it doesn't exist
        os.makedirs("src/table_pose", exist_ok=True)
        
        # Save the calibration points
        np.savez("src/table_pose/right.npz", left_table_point=self.pixel2points['left'])
        np.savez("src/table_pose/left.npz", right_table_point=self.pixel2points['right'])
        
        # Use points from both cameras to calculate transformation
        left_points = self.points['left']
        right_points = self.points['right']
        result = []
        
        # Calculate 3D coordinates for each point pair
        # for i in range(len(left_points)):
        #     x1, x2 = np.array(left_points[i]), np.array(right_points[i])
        #     A = np.array([[x1[0], -self.R_cam[0, 0] * x2[0] - self.R_cam[0, 1] * x2[1] - self.R_cam[0, 2]],
        #                  [x1[1], -self.R_cam[1, 0] * x2[0] - self.R_cam[1, 1] * x2[1] - self.R_cam[1, 2]],
        #                  [1, -self.R_cam[2, 0] * x2[0] - self.R_cam[2, 1] * x2[1] - self.R_cam[2, 2]]])
        #     B = self.T_cam.flatten()
        #     depths = np.linalg.lstsq(A, B, rcond=None)[0]
        #     d1 = depths[0]
        #     X = d1 * x1
        #     result.append(X)
        for i in range(len(left_points)):
            # print(np.array(left_points[i][:2]), np.array(right_points[i][:2]))
            result.append(self.triangulator.process_frames_thread(np.array(left_points[i][:2]), np.array(right_points[i][:2])))
            print('--------------------------------')

        print(result)

        # Calculate basis vectors and ensure orthogonality using Gram-Schmidt process
        vx = result[1] - result[0]  # Vector from point 0 to point 1 (width)
        vy = result[3] - result[0]  # Vector from point 0 to point 3 (length)
        
        # Normalize x-axis
        ux = vx / np.linalg.norm(vx)
        
        # Make y-axis orthogonal to x-axis using Gram-Schmidt
        vy_proj = np.dot(vy, ux) * ux  # Project vy onto ux
        vy_orth = vy - vy_proj  # Make vy orthogonal to ux
        uy = vy_orth / np.linalg.norm(vy_orth)  # Normalize
        
        # Calculate z-axis as cross product (ensures right-hand coordinate system)
        uz = np.cross(ux, uy)
        uz = uz / np.linalg.norm(uz)  # Normalize

        # Construct the rotation matrix
        R = np.array([ux, uy, uz]).T
        T = np.array(result[0]).reshape((3, 1))

        # Save transformation parameters
        np.savez("src/pose_parameters.npz", rotation_matrix=R, translation_matrix=T)
        
        # Print verification and transformed points
        print("\nTransformation calculated successfully!")
        print("Orthogonality check:", np.allclose(R @ R.T, np.eye(3), atol=1e-6))
        print("Determinant (should be close to 1):", np.linalg.det(R))
        
        print("\nTransformed corner points (in table coordinates, centimeters):")
        for i, point in enumerate(result):
            point_cam = np.array(point).reshape((3, 1))
            transformed_point = np.round(R.T @ (point_cam - T))/10
            print(f"Corner {i+1}:", transformed_point.flatten())
        
        return R, T

def main():
    calibrator = TableCalibration()
    print("We'll calibrate both cameras. Please mark the 4 corners of the table tennis table in this order:")
    print("1. Top-left corner")
    print("2. Top-right corner")
    print("3. Bottom-left corner")
    print("4. Bottom-right corner")
    print("\nPress 'r' to reset points if you make a mistake")
    print("Press 'c' to continue after marking all 4 points")
    print("Press 'q' to quit\n")
    
    # Capture corners for left camera
    print("\nStarting with LEFT camera...")
    if not calibrator.capture_corners("camera_test/left/img1.png", 'left'):
        print("\nCalibration cancelled.")
        return
        
    # Capture corners for right camera
    print("\nNow for RIGHT camera...")
    if not calibrator.capture_corners("camera_test/right/img1.png", 'right'):
        print("\nCalibration cancelled.")
        return
    
    # Calculate transformation using both sets of points
    R, T = calibrator.calculate_transformation()
    print("\nCalibration complete! Transformation matrices saved to 'pose_parameters.npz'")
    print("Corner points saved to 'table_pose/left.npz' and 'table_pose/right.npz'")

if __name__ == "__main__":
    main()
