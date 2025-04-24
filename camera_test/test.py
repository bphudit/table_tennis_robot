import time
from camera_manager_instantarray import CameraManager
from serial_control import SerialController
from visualization import FrameVisualizer
import numpy as np
from ball_triangulation_cv import BallTriangulatorP
import cv2
import cv2.aruco as aruco
import json
from scipy.io import loadmat

with open('hsv_values.json', 'r') as f:
    hsv_values = json.load(f)

stereo_params = loadmat('src/stereoParameters.mat')
pose_params = np.load('src/pose_parameters.npz')

def main():
    # Initialize components
    serial_controller = SerialController(port='/dev/cu.usbserial-0001', baudrate=115200)
    camera_manager = CameraManager(exposure_time=6000, gain=10, mode=0)
    visualizer = FrameVisualizer(scale_factor=0.5)
    ball_triangulator_cv = BallTriangulatorP(stereo_params, pose_params=None, coord_frame='camera')

    count = 0  # Counter for synchronization errors

    # Define dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    left_points = []
    right_points = []

    result = []
    try:
        print("Press 'q' to quit")
        serial_controller.start_trigger()
        
        while True:
            # Get synchronized frames and measure processing time
            frame_start_time = time.time()
            img1, img2, block_id1, block_id2 = camera_manager.get_synchronized_frames()

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Process frames if available
            if img1 is not None and img2 is not None:
                
                # Calculate processing time
                frame_processing_time = (time.time() - frame_start_time) * 1000  # ms
                
                # Display frames with stats
                # key = visualizer.display_frames(img1, img2, frame_processing_time)

                # Detect markers
                corners1, ids1, rejected1 = detector.detectMarkers(gray1)
                corners2, ids2, rejected2 = detector.detectMarkers(gray2)

                for i in range(len(ids1)):
                    if ids1[i] == 971:

                        bottom_left_x = corners1[i][0][0][0]
                        bottom_left_y = corners1[i][0][0][1]

                        top_left_x = corners1[i][0][1][0]
                        top_left_y = corners1[i][0][1][1]

                        cv2.circle(gray1, (int(bottom_left_x), int(bottom_left_y)), 10, (0, 255, 0), -1)
                        cv2.circle(gray1, (int(top_left_x), int(top_left_y)), 10, (0, 0, 255), -1)

                        left_points.append([bottom_left_x, bottom_left_y])
                        left_points.append([top_left_x, top_left_y])
                        print('a')
                    elif ids1[i] == 212: 

                        top_right_x = corners1[i][0][2][0]
                        top_right_y = corners1[i][0][2][1]

                        bottom_right_x = corners1[i][0][3][0]
                        bottom_right_y = corners1[i][0][3][1]

                        cv2.circle(gray1, (int(top_right_x), int(top_right_y)), 10, (0, 255, 0), -1)
                        cv2.circle(gray1, (int(bottom_right_x), int(bottom_right_y)), 10, (0, 0, 255), -1)

                        left_points.append([bottom_right_x, bottom_right_y])
                        left_points.append([top_right_x, top_right_y])
                        print('b')
                for i in range(len(ids2)):
                    if ids2[i] == 971:
                        bottom_left_x = corners2[i][0][0][0]
                        bottom_left_y = corners2[i][0][0][1]

                        top_left_x = corners2[i][0][1][0]
                        top_left_y = corners2[i][0][1][1]

                        cv2.circle(gray2, (int(bottom_left_x), int(bottom_left_y)), 10, (0, 255, 0), -1)
                        cv2.circle(gray2, (int(top_left_x), int(top_left_y)), 10, (0, 0, 255), -1)

                        right_points.append([bottom_left_x, bottom_left_y])
                        right_points.append([top_left_x, top_left_y])
                        print('c')
                    elif ids2[i] == 212: 

                        top_right_x = corners2[i][0][2][0]
                        top_right_y = corners2[i][0][2][1]

                        bottom_right_x = corners2[i][0][3][0]
                        bottom_right_y = corners2[i][0][3][1]

                        cv2.circle(gray2, (int(top_right_x), int(top_right_y)), 10, (0, 255, 0), -1)
                        cv2.circle(gray2, (int(bottom_right_x), int(bottom_right_y)), 10, (0, 0, 255), -1)

                        right_points.append([top_right_x, top_right_y])
                        right_points.append([bottom_right_x, bottom_right_y])
                        print('d')



                for i in range(len(left_points)):
                    result.append(ball_triangulator_cv.process_frames_thread(np.array(left_points[i][:2]), np.array(right_points[i][:2])))

                if len(result) == 4:
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




                # cv2.imshow("Left Camera", gray1)
                # cv2.imshow("Right Camera", gray2)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Handle synchronization errors
                print("No synchronized frames available")
                print(f"Block ID 1: {block_id1} != Block ID 2: {block_id2}")
                count += 1
                print(f"Count: {count}")

        
                

                
    finally:
        # Clean up resources
        serial_controller.stop_trigger()
        camera_manager.close()
        visualizer.close()
        

if __name__ == "__main__":
    main() 