import time
from main.camera_manager_instantarray import CameraManager
from main.serial_control import SerialController
from main.visualization import FrameVisualizer
from main.ball_triangulation_cv import BallTriangulatorP
import numpy as np
import cv2
import json
from scipy.io import loadmat

with open('hsv_values.json', 'r') as f:
    hsv_values = json.load(f)

stereo_params = loadmat('src/stereoParameters.mat')
pose_params = np.load('src/pose_parameters.npz')

def main():
    # Initialize components
    serial_controller = SerialController(port='/dev/cu.usbserial-0001', baudrate=115200)
    camera_manager = CameraManager(exposure_time=20000, gain=0, mode=1)
    visualizer = FrameVisualizer(scale_factor=0.5)
    ball_triangulator_cv = BallTriangulatorP(stereo_params, pose_params=None, coord_frame='camera')

    count = 0  # Counter for synchronization errors

    # Define dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Initialize arrays with None values to store points in correct order
    left_points = [None] * 4  # [a, b, c, d]
    right_points = [None] * 4  # [e, f, g, h]

    result = []
    try:
        print("Press 'q' to quit")
        serial_controller.start_trigger()
        
        while True:
            # Get synchronized frames and measure processing time
            frame_start_time = time.time()
            img1, img2, bid, tms = camera_manager.get_synchronized_frames()

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Process frames if available
            if img1 is not None and img2 is not None:
                
                # # Calculate processing 
                # frame_processing_time = (time.time() - frame_start_time) * 1000  # ms
                
                # # Display frames with stats
                # # key = visualizer.display_frames(img1, img2, frame_processing_time)

                # Detect markers
                corners1, ids1, rejected1 = detector.detectMarkers(gray1)
                corners2, ids2, rejected2 = detector.detectMarkers(gray2)
                # # Draw detected markers on both frames
                # if ids1 is not None:
                #     cv2.aruco.drawDetectedMarkers(gray1, corners1, ids1)
                # if ids2 is not None:
                #     cv2.aruco.drawDetectedMarkers(gray2, corners2, ids2)

                print(ids1)
                print(ids2)
                print('--------------------------------')        

                if ids1 is not None and ids2 is not None:
                    if len(ids1) == 4 and len(ids2) == 4:
                        # Reset points arrays for each new complete detection
                        left_points = [None, None, None, None]
                        right_points = [None, None, None, None]
                        
                        # Process left camera markers (a, b, c, d)
                        for i in range(len(ids1)):
                            if ids1[i] == 0:
                                bottom_left_x = corners1[i][0][0][0]
                                bottom_left_y = corners1[i][0][0][1]
                                cv2.circle(gray1, (int(bottom_left_x), int(bottom_left_y)), 10, (0, 255, 0), -1)
                                
                                # Store at index 0 (a)
                                left_points[0] = [bottom_left_x, bottom_left_y]
                            elif ids1[i] == 1: 
                                top_left_x = corners1[i][0][1][0]
                                top_left_y = corners1[i][0][1][1]
                                cv2.circle(gray1, (int(top_left_x), int(top_left_y)), 10, (0, 0, 255), -1)
                                
                                # Store at index 1 (b)
                                left_points[1] = [top_left_x, top_left_y]
                            elif ids1[i] == 2:
                                top_right_x = corners1[i][0][2][0]
                                top_right_y = corners1[i][0][2][1]
                                cv2.circle(gray1, (int(top_right_x), int(top_right_y)), 10, (0, 255, 0), -1)
                                
                                # Store at index 2 (c)
                                left_points[2] = [top_right_x, top_right_y]
                            elif ids1[i] == 3:
                                bottom_right_x = corners1[i][0][3][0]
                                bottom_right_y = corners1[i][0][3][1]
                                cv2.circle(gray1, (int(bottom_right_x), int(bottom_right_y)), 10, (0, 0, 255), -1)
                                
                                # Store at index 3 (d)
                                left_points[3] = [bottom_right_x, bottom_right_y]
                        
                        # Process right camera markers (e, f, g, h)
                        for i in range(len(ids2)):
                            if ids2[i] == 0:
                                bottom_left_x = corners2[i][0][0][0]
                                bottom_left_y = corners2[i][0][0][1]
                                cv2.circle(gray2, (int(bottom_left_x), int(bottom_left_y)), 10, (0, 255, 0), -1)
                                
                                # Store at index 0 (e)
                                right_points[0] = [bottom_left_x, bottom_left_y]
                            elif ids2[i] == 1: 
                                top_left_x = corners2[i][0][1][0]
                                top_left_y = corners2[i][0][1][1]
                                cv2.circle(gray2, (int(top_left_x), int(top_left_y)), 10, (0, 0, 255), -1)
                                
                                # Store at index 1 (f)
                                right_points[1] = [top_left_x, top_left_y]
                            elif ids2[i] == 2:
                                top_right_x = corners2[i][0][2][0]
                                top_right_y = corners2[i][0][2][1]
                                cv2.circle(gray2, (int(top_right_x), int(top_right_y)), 10, (0, 255, 0), -1)
                                
                                # Store at index 2 (g)
                                right_points[2] = [top_right_x, top_right_y]
                            elif ids2[i] == 3:
                                bottom_right_x = corners2[i][0][3][0]
                                bottom_right_y = corners2[i][0][3][1]
                                cv2.circle(gray2, (int(bottom_right_x), int(bottom_right_y)), 10, (0, 0, 255), -1)
                                
                                # Store at index 3 (h)
                                right_points[3] = [bottom_right_x, bottom_right_y]

                        # Check if all markers were detected correctly in both cameras
                        if None not in left_points and None not in right_points:
                            # Clear previous results
                            result = []
                            # Process triangulation for all points in order
                            for i in range(4):
                                result.append(ball_triangulator_cv.process_frames_thread(
                                    np.array(left_points[i]), 
                                    np.array(right_points[i]),
                                    tms
                                )[:3])
                            print("All points processed in order!")
                            # print(result)
                        else:
                            print("Not all markers detected correctly")
                            left_points = [None, None, None, None]
                            right_points = [None, None, None, None]

                if len(result) == 4:
                    # Use point 0 as the origin
                    origin = result[0]
                    T = np.array(origin).reshape((3, 1))
                    
                    # Shift all points to be relative to the origin
                    points_relative = [result[i] - origin for i in range(4)]
                    
                    # Fit a plane to all 4 points using least squares
                    # We want to find normal vector to the plane (will be our z-axis)
                    # Create a matrix of the x and y coordinates
                    A = np.array([[p[0], p[1], 1] for p in points_relative])
                    b = np.array([p[2] for p in points_relative])
                    
                    # Solve for the plane equation: ax + by + c = z
                    # This gives us the normal vector [a, b, -1]
                    plane_params, _, _, _ = np.linalg.lstsq(A[:, :2], b, rcond=None)
                    normal = np.array([plane_params[0], plane_params[1], -1])
                    uz = -normal / np.linalg.norm(normal)  # Normalize and flip to point upward
                    
                    # Calculate initial x-axis (from point 0 to point 1)
                    vx_initial = result[1] - origin
                    
                    # Project vx_initial onto the plane (remove component along normal)
                    vx = vx_initial - np.dot(vx_initial, uz) * uz
                    ux = vx / np.linalg.norm(vx)  # Normalize
                    
                    # Calculate y-axis as cross product to ensure orthogonality (right-handed system)
                    uy = np.cross(uz, ux)
                    uy = uy / np.linalg.norm(uy)  # Normalize
                    
                    # Double-check orthogonality with another cross product
                    uz_check = np.cross(ux, uy)
                    uz = uz_check  # This ensures perfect orthogonality
                    
                    # Construct the rotation matrix
                    R = np.array([ux, uy, uz]).T
                    
                    # Save transformation parameters
                    np.savez("src/pose_parameters.npz", rotation_matrix=R, translation_matrix=T)
                    
                    # Print verification and transformed points
                    print("\nTransformation calculated successfully!")
                    print("Orthogonality check:", np.allclose(R @ R.T, np.eye(3), atol=1e-6))
                    print("Determinant (should be close to 1):", np.linalg.det(R))
                    
                    # Calculate the distance of each point from the fitted plane
                    print("\nDistance of each point from the fitted plane (mm):")
                    for i, point in enumerate(result):
                        point_relative = point - origin
                        distance = np.abs(np.dot(point_relative, uz))
                        print(f"Corner {i}: {distance*10:.2f} mm")
                    
                    print("\nTransformed corner points (in table coordinates, centimeters):")
                    for i, point in enumerate(result):
                        point_cam = np.array(point).reshape((3, 1))
                        transformed_point = np.round(R.T @ (point_cam - T))/10
                        print(f"Corner {i+1}:", transformed_point.flatten())
                    
                    return R, T


                cv2.imshow("Left Camera", gray1)
                cv2.imshow("Right Camera", gray2)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Handle synchronization errors
                print("No synchronized frames available")
    
    finally:
        # Clean up resources
        serial_controller.stop_trigger()
        camera_manager.close()
        visualizer.close()
        

if __name__ == "__main__":
    main() 