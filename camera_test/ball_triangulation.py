import numpy as np
import time

class BallTriangulator:
    def __init__(self, stereo_params, pose_params):

        # Camera intrinsics and extrinsics
        self.K1 = stereo_params['K1']
        self.K2 = stereo_params['K2']
        self.R = stereo_params['R']
        self.T = stereo_params['T'].reshape((3, 1))
        
        # Invert for camera 2 to camera 1 transformation
        self.R = np.linalg.inv(self.R)
        self.T = -self.R @ self.T
        
        # Inverse of intrinsic matrices
        self.K1_inv = np.linalg.inv(self.K1)
        self.K2_inv = np.linalg.inv(self.K2)
        
        # Transformation to table coordinate system
        self.pose_rotation = pose_params['rotation_matrix']
        self.pose_translation = pose_params['translation_matrix']
        
        # 3D position data
        self.position_3D = []
    
    def process_frames_thread(self, coords_left, coords_right):
        # If we have detected the ball in both frames, triangulate
        if coords_left and coords_right:
            u1, v1 = coords_left
            u2, v2 = coords_right

            
            # Convert image coordinates to normalized coordinates
            x1 = self.K1_inv @ np.array([u1, v1, 1])
            x2 = self.K2_inv @ np.array([u2, v2, 1])
            
            # Triangulate position
            position = self.triangulate_position(x1, x2)
            if position is not None:
                print(f"Ball position: {position}")
                
    
    
    def triangulate_position(self, x1, x2):
        """
        Triangulate the 3D position of the ball based on stereo camera views
        
        Args:
            x1: Normalized coordinates from left camera
            x2: Normalized coordinates from right camera
        
        Returns:
            X: 3D position in table coordinates [x, y, z]
            Or None if triangulation is not possible
        """
        # Set up and solve for depths d1 and d2
        A = np.array([
            [x1[0], -self.R[0, 0] * x2[0] - self.R[0, 1] * x2[1] - self.R[0, 2]],
            [x1[1], -self.R[1, 0] * x2[0] - self.R[1, 1] * x2[1] - self.R[1, 2]],
            [1, -self.R[2, 0] * x2[0] - self.R[2, 1] * x2[1] - self.R[2, 2]]
        ])
        B = self.T.flatten()
        
        # Solve for d1 and d2
        depths = np.linalg.lstsq(A, B, rcond=None)[0]
        d1 = depths[0]
        
        # Compute the 3D point in Camera 1's frame
        X = d1 * x1
        
        # Transform to table coordinate system
        # X = self.pose_rotation.T @ (X_cam.reshape((3, 1)) - self.pose_translation)
        
        # Store timestamp with position
        t = time.time()
        x, y, z = float(X[0]), float(X[1]), float(-X[2] - 40)  # Match the convention in depth_test_thread.py
        
        # Record position with timestamp
        self.position_3D.append([x, y, z, t])
        
        return np.array([x, y, z])
    



