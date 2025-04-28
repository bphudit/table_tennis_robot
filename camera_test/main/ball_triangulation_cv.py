import cv2
import numpy as np
import time

class BallTriangulatorP:
    def __init__(self, stereo_params, pose_params=None, coord_frame = 'table'):
        # Camera intrinsics and extrinsics
        self.K1 = stereo_params['K1']  # 3x3 intrinsic for cam1
        self.K2 = stereo_params['K2']  # 3x3 intrinsic for cam2
        self.R = stereo_params['R']    # 3x3 rotation from cam1 to cam2
        self.T = stereo_params['T'].reshape((3, 1))  # 3x1 translation from cam1 to cam2

        # Build projection matrices for cv2.triangulatePoints
        # P1 = K1 [I | 0]
        self.P1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        # P2 = K2 [R | T]
        self.P2 = self.K2 @ np.hstack((self.R, self.T))

        # Store distortion if available (optional)
        self.D1 = stereo_params.get('D1', None)
        self.D2 = stereo_params.get('D2', None)

        if pose_params is not None:
            # Pose parameters for table coordinate transform
            self.pose_rotation = pose_params['rotation_matrix']      # 3x3
            self.pose_translation = pose_params['translation_matrix'] # 3x1
        self.coord_frame = coord_frame

        # Preallocate 3D position log
        self.position_3D = []

    def process_frames_thread(self, coords_left, coords_right, tms):
        # Proceed only if both detections exist
        if coords_left is not None and coords_right is not None:
            u1, v1 = coords_left
            u2, v2 = coords_right

            # Triangulate using OpenCV's DLT implementation
            pts1 = np.array([[u1], [v1]], dtype=np.float64)
            pts2 = np.array([[u2], [v2]], dtype=np.float64)

            # cv2.triangulatePoints expects shape (3x4) projection mats and 2xN point arrays
            homogeneous_4d = cv2.triangulatePoints(self.P1, self.P2, pts1, pts2)

            # Convert from homogeneous to 3D (divide by last row)
            points_3d = homogeneous_4d[:3, 0] / homogeneous_4d[3, 0]
            X_cam = points_3d.reshape((3, 1))

            if self.coord_frame == 'table':
                # Transform to table coordinate system
                X_table = self.pose_rotation.T @ (X_cam - self.pose_translation)

                # Optionally apply any convention shifts (e.g., -40mm offset)
                x, y, z = float(X_table[0]), float(X_table[1]), float(X_table[2])
            elif self.coord_frame == 'camera':
                x, y, z = float(X_cam[0]), float(X_cam[1]), float(X_cam[2])

            # Timestamp
            t = tms
            self.position_3D.append([x, y, z, t])

            print(f"Ball position: [{x:.2f}, {y:.2f}, {z:.2f}] at {t:.3f}")

            return np.array([x, y, z, t])
        return None