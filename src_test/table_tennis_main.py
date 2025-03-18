import numpy as np
import cv2
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Import from utility modules
from src.vision_utils import (
    init_stereo_cameras, 
    start_ball_tracking, 
    calculate_3d_position, 
    cleanup_resources
)
from src.kalman_utils import (
    init_KF,
    calculate_velocity,
    update_filter_with_measurement,
    predict_trajectory,
    calculate_hitting_position,
    determine_hit_type
)
from src.robot_utils import RobotController

def setup_calibration():
    """Load and prepare calibration data."""
    data = loadmat('stereoParameters.mat')
    data2 = np.load('pose_parameters.npz')

    # Extract calibration parameters
    pose_rotation = data2['rotation_matrix']
    pose_translation = data2['translation_matrix']
    K1 = data['K1']  # Left camera intrinsic matrix
    K2 = data['K2']  # Right camera intrinsic matrix
    R = data['R']    # Rotation matrix between cameras
    T = data['T'].reshape((3, 1))  # Translation vector between cameras

    # Prepare matrices for 3D reconstruction
    R = np.linalg.inv(R)
    T = -R @ T
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    
    return K1_inv, K2_inv, R, T, pose_rotation, pose_translation

def plot_results(position_3D, prediction_x, prediction_y, prediction_z, 
                pure_prediction_x, pure_prediction_y, pure_prediction_z):
    """Plot the tracking and prediction results."""
    if len(position_3D) > 0:
        position_3D = np.array(position_3D)
        ax = plt.axes(projection='3d')

        ax.set_xlim([1525, 0])
        ax.set_ylim([0, 2740])
        ax.set_zlim([0, 1500])

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('height [mm]')

        # Plot measured positions
        ax.scatter(position_3D[:, 0], position_3D[:, 1], position_3D[:, 2], 
                marker='o', alpha=1, color='r', label='Data from Sensor')
        
        # Plot Kalman filter predictions during tracking
        if prediction_x:
            ax.scatter(prediction_x, prediction_y, prediction_z, 
                    marker='o', alpha=1, color='b', label='KF prediction')
        
        # Plot pure predictions (future trajectory)
        if pure_prediction_x:
            ax.scatter(pure_prediction_x, pure_prediction_y, pure_prediction_z, 
                    marker='o', alpha=1, color='g', label='KF pure prediction')

        ax.legend()
        plt.show()

def main():
    # 1. Load calibration data
    print("Loading calibration data...")
    K1_inv, K2_inv, R, T, pose_rotation, pose_translation = setup_calibration()

    # 2. Set up color thresholds for ball detection
    left_color_limits = (np.array([0, 212, 141]), np.array([81, 255, 255]))
    right_color_limits = (np.array([0, 109, 151]), np.array([81, 255, 255]))

    # 3. Initialize cameras
    print("Initializing cameras...")
    left_camera, right_camera, left_frame_queue, right_frame_queue, left_pos_queue, right_pos_queue, stop_event = init_stereo_cameras(
        left_color_limits=left_color_limits,
        right_color_limits=right_color_limits,
        exposure_time=10000,
        gain=0
    )

    # 4. Start ball tracking
    print("Starting ball tracking...")
    left_tracker, right_tracker = start_ball_tracking(
        left_camera, right_camera, left_frame_queue, right_frame_queue,
        left_pos_queue, right_pos_queue, K1_inv, K2_inv, stop_event,
        left_color_limits=left_color_limits,
        right_color_limits=right_color_limits
    )

    # 5. Initialize robot
    print("Initializing robot...")
    robot = RobotController()
    robot.stand_by()
    print("Robot in standby position")

    # 6. Data storage
    position_3D = []
    position_times = []
    prediction_x = []
    prediction_y = []
    prediction_z = []
    pure_prediction_x = []
    pure_prediction_y = []
    pure_prediction_z = []
    kf = None

    # 7. Main tracking loop
    try:
        print("Starting main tracking loop...")
        while True:
            # Display camera frames
            if not left_frame_queue.empty():
                left_frame = left_frame_queue.get()
                cv2.imshow("Left Camera", left_frame)

            if not right_frame_queue.empty():
                right_frame = right_frame_queue.get()
                cv2.imshow("Right Camera", right_frame)

            # Process ball positions if detected in both cameras
            if not left_pos_queue.empty() and not right_pos_queue.empty():
                x1 = left_pos_queue.get()
                x2 = right_pos_queue.get()

                # Calculate 3D position
                X = calculate_3d_position(x1, x2, R, T, pose_rotation, pose_translation)
                t = time.time()

                # Transform to table coordinates
                x, y, z = float(X[0]), float(X[1]), float(-X[2] - 40)
                
                # Store position and time
                position_3D.append([x, y, z])
                position_times.append(t)

                if y >= 2740 // 2:  # Ball is still on opponent's side
                    if len(position_3D) == 2:
                        # Initialize Kalman filter with first velocity estimate
                        velocity = calculate_velocity(position_3D, position_times)
                        initial_val = np.array([[x], [y], [z], 
                                              [velocity[0]], [velocity[1]], [velocity[2]]])
                        kf = init_KF(measurements=initial_val)
                        
                    elif len(position_3D) > 2:
                        # Update Kalman filter with new measurement
                        velocity = calculate_velocity(position_3D, position_times)
                        dt = position_times[-1] - position_times[-2]
                        
                        kf = update_filter_with_measurement(
                            kf, 
                            [x, y, z], 
                            velocity, 
                            dt
                        )
                        
                        # Store predicted position for visualization
                        prediction_x.append(kf.x[0][0])
                        prediction_y.append(kf.x[1][0])
                        prediction_z.append(kf.x[2][0])
                else:
                    # Ball is approaching robot side, make final prediction and hit
                    print("Ball approaching robot side, making final prediction...")
                    
                    # Stop camera threads
                    stop_event.set()
                    
                    # Predict trajectory until ball crosses table edge
                    if kf is not None:
                        (pred_x, pred_y, pred_z), final_state = predict_trajectory(kf)
                        
                        # Store prediction for visualization
                        pure_prediction_x = pred_x
                        pure_prediction_y = pred_y
                        pure_prediction_z = pred_z
                        
                        # Calculate robot hitting position
                        robot_x, robot_z = calculate_hitting_position(final_state)
                        
                        # Determine hit type
                        hit_type = determine_hit_type(robot_x, robot_z)
                        
                        print(f"Hit position calculated: x={robot_x:.1f}, z={robot_z:.1f}, type={hit_type}")
                        
                        # Execute appropriate hitting motion
                        if hit_type != "none":
                            print(f"Executing {hit_type} hit")
                            robot.execute_hit(hit_type, robot_x, robot_z)
                        else:
                            print("Target out of range, not hitting")
                    else:
                        print("No Kalman filter initialized, cannot predict")
                    
                    break

                print(f"3D Position: {X.flatten()}")

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        print("Cleaning up resources...")
        
        # Clean up robot
        robot.close()
        
        # Wait for tracker threads to finish
        left_tracker.join()
        right_tracker.join()
        
        # Clean up camera resources
        cleanup_resources(left_camera, right_camera, left_tracker, right_tracker, stop_event)
        
        # Plot tracking results
        plot_results(
            position_3D, prediction_x, prediction_y, prediction_z,
            pure_prediction_x, pure_prediction_y, pure_prediction_z
        )
        
        # Optionally save data
        if position_3D:
            position_3D_array = np.array(position_3D)
            np.savez("data/tracking_data.npz", position_3D=position_3D_array)
            print("Data saved to data/tracking_data.npz")

if __name__ == "__main__":
    main() 