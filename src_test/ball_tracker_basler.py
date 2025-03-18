import numpy as np
import cv2
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
from threading import Event

# Import from local modules
from vision_utils import (
    init_stereo_cameras, 
    start_ball_tracking, 
    calculate_3d_position, 
    cleanup_resources
)
from kalman_utils import KalmanFilter
import URController.URBasic as URBasic

def init_KF(measurements):
    """Initialize Kalman Filter with appropriate parameters"""
    initial_state = measurements

    initial_cov = np.array([
        [2, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 5, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    # Process Noise (Lower = trust more in model)
    Q = np.array([
        [0.1, 0,   0,   0,   0,   0],   # Variance for Px
        [0,   0.1, 0,   0,   0,   0],   # Variance for Py
        [0,   0,   2,   0,   0,   0],   # Variance for Pz (more uncertainty)
        [0,   0,   0,   0.5, 0,   0],   # Variance for Vx
        [0,   0,   0,   0,   0.5, 0],   # Variance for Vy
        [0,   0,   0,   0,   0,   5]    # Variance for Vz (more uncertainty)
    ])

    # Measurement Noise
    R = np.array([
        [1.0, 0,   0,   0,   0,   0],   # Variance for measured Px
        [0,   1.0, 0,   0,   0,   0],   # Variance for measured Py
        [0,   0,   0.5, 0,   0,   0],   # Variance for measured Pz
        [0,   0,   0,   5.0, 0,   0],   # Variance for measured Vx
        [0,   0,   0,   0,   5.0, 0],   # Variance for measured Vy
        [0,   0,   0,   0,   0,   2.0]  # Variance for measured Vz
    ])

    kf = KalmanFilter(
        process_noise_std=Q, 
        measurement_noise_std=R, 
        initial_state=initial_state, 
        initial_covariance=initial_cov, 
        restitution_coeff=0.8
    )
    return kf

class RobotController:
    """UR Robot controller class"""
    def __init__(self, robot_ip='10.10.0.7'):
        self.robot = URBasic.robotModel.RobotModel()
        self.ur = URBasic.urScriptExt.UrScriptExt(host=robot_ip, robotModel=self.robot)
        
    def fore_hand(self, x, z):
        """Forehand hitting position"""
        if x >= 0:
            z = max(40, z)
            target_pose_fore = [x/1000, 300/1000, z/1000, 1.240, 1.168, 1.306]
            self.ur.movej(pose=target_pose_fore, a=100, v=100)
        else:
            print("X IS LESS THAN 0")

    def stand_by(self):
        """Standby position"""
        target_pose_standby = [91/1000, 625/1000, 418/1000, 0.125, -1.71, -2.639]
        self.ur.movej(pose=target_pose_standby, a=2, v=2)

    def back_hand(self, x, z):
        """Backhand hitting position"""
        if x < 0:
            z = max(40, z)
            target_pose_back = [x/1000, 300/1000, z/1000, 1.206, -1.217, -1.187]
            self.ur.movej(pose=target_pose_back, a=100, v=100)
        else:
            print('X IS MORE THAN 0')
    
    def close(self):
        """Close the robot connection"""
        self.ur.close()

def plot_trajectories(position_3D, prediction_x, prediction_y, prediction_z, 
                     pure_prediction_x, pure_prediction_y, pure_prediction_z):
    """Plot measured and predicted trajectories in 3D"""
    position_3D = np.array(position_3D)
    ax = plt.axes(projection='3d')

    ax.set_xlim([1525, 0])
    ax.set_ylim([0, 2740])
    ax.set_zlim([0, 1500])

    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('height [mm]')

    ax.scatter(position_3D[:, 0], position_3D[:, 1], position_3D[:, 2], 
              marker='o', alpha=1, color='r', label='Data from Sensor')
    
    ax.scatter(prediction_x, prediction_y, prediction_z, 
              marker='o', alpha=1, color='b', label='KF prediction')
    
    ax.scatter(pure_prediction_x, pure_prediction_y, pure_prediction_z, 
              marker='o', alpha=1, color='g', label='KF pure prediction')

    ax.legend()
    plt.show()

def main():
    # Load calibration data
    data = loadmat('src\stereoParameters.mat')
    data2 = np.load('src\pose_parameters.npz')

    pose_rotation = data2['rotation_matrix']
    pose_translation = data2['translation_matrix']

    # Intrinsic matrices
    K1 = data['K1']
    K2 = data['K2']

    # Rotation and translation from Camera 2 to Camera 1
    R = data['R']
    T = data['T'].reshape((3, 1))

    R = np.linalg.inv(R)
    T = -R @ T

    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    # Color thresholds for ball detection
    left_color_limits = (np.array([0, 100, 100]), np.array([80, 255, 255]))
    right_color_limits = (np.array([0, 109, 151]), np.array([81, 255, 255]))

    # Initialize stereo cameras
    print("Initializing cameras...")
    left_camera, right_camera, left_frame_queue, right_frame_queue, left_pos_queue, right_pos_queue, left_mask_queue, right_mask_queue, stop_event = init_stereo_cameras(
        left_color_limits=left_color_limits,
        right_color_limits=right_color_limits,
        exposure_time=6000,
        gain=0
    )

    # Start ball tracking
    print("Starting ball tracking...")
    left_tracker, right_tracker = start_ball_tracking(
        left_camera, right_camera, left_frame_queue, right_frame_queue,
        left_pos_queue, right_pos_queue, K1_inv, K2_inv, stop_event,
        left_mask_queue=left_mask_queue, right_mask_queue=right_mask_queue,
        left_color_limits=left_color_limits,
        right_color_limits=right_color_limits
    )

    # Initialize robot
    # print("Initializing robot...")
    # robot = RobotController()
    # robot.stand_by()

    # Data storage
    position_3D = []
    num_data = 0
    prediction_x = []
    prediction_y = []
    prediction_z = []
    pure_prediction_x = []
    pure_prediction_y = []
    pure_prediction_z = []
    kf = None

    try:
        print("Main tracking loop starting...")
        while True:
            # Display processed frames
            if not left_frame_queue.empty():
                left_frame = left_frame_queue.get()
                cv2.imshow("Left Camera", left_frame)

            if not right_frame_queue.empty():
                right_frame = right_frame_queue.get()
                cv2.imshow("Right Camera", right_frame)
                
            # Display HSV masks for debugging
            if not left_mask_queue.empty():
                left_mask = left_mask_queue.get()
                cv2.imshow("Left Camera HSV Mask", left_mask)
                
            if not right_mask_queue.empty():
                right_mask = right_mask_queue.get()
                cv2.imshow("Right Camera HSV Mask", right_mask)

            # Process ball positions if detected in both cameras
            if not left_pos_queue.empty() and not right_pos_queue.empty():
                x1 = left_pos_queue.get()
                x2 = right_pos_queue.get()

                # Calculate 3D position
                X = calculate_3d_position(x1, x2, R, T, pose_rotation, pose_translation)
                t = time.time()

                # Transform to table coordinates
                x, y, z = float(X[0]), float(X[1]), float(-X[2] - 40)
                position_3D.append([x, y, z, t])

                # if y >= 2740 // 2:  # Ball is still far away
                #     if num_data == 1:
                #         # Initialize Kalman filter with first velocity estimate
                #         dt = position_3D[num_data][-1] - position_3D[num_data - 1][-1]
                #         dt = max(dt, 0.00001)

                #         velocity_x = (position_3D[num_data][0] - position_3D[num_data - 1][0]) / dt
                #         velocity_y = (position_3D[num_data][1] - position_3D[num_data - 1][1]) / dt
                #         velocity_z = (position_3D[num_data][2] - position_3D[num_data - 1][2]) / dt

                #         initial_val = np.array([[x], [y], [z], [velocity_x], [velocity_y], [velocity_z]])
                #         kf = init_KF(measurements=initial_val)
                #         kf.predict(dt)

                #     elif num_data > 1:
                #         # Update Kalman filter with measurements
                #         dt = position_3D[num_data][-1] - position_3D[num_data - 1][-1]
                #         dt = max(dt, 0.00001)

                #         velocity_x = (position_3D[num_data][0] - position_3D[num_data - 1][0]) / dt
                #         velocity_y = (position_3D[num_data][1] - position_3D[num_data - 1][1]) / dt
                #         velocity_z = (position_3D[num_data][2] - position_3D[num_data - 1][2]) / dt

                #         kf.update(z=np.array([[x], [y], [z], [velocity_x], [velocity_y], [velocity_z]]))
                #         kf.predict(dt)

                #         if kf.x[2] <= 0:
                #             kf.handle_bounce()

                #         prediction_x.append(kf.x[0])
                #         prediction_y.append(kf.x[1])
                #         prediction_z.append(kf.x[2])
                # else:
                #     # Ball is approaching robot side, make final prediction and hit
                #     print("Ball approaching robot side, making final prediction...")
                    
                #     # Stop camera threads
                #     stop_event.set()
                #     left_tracker.join()
                #     right_tracker.join()

                #     # Make final trajectory prediction
                #     dt = 0.01
                #     while kf.x[1] > 0:
                #         kf.predict(dt)
                #         if kf.x[2] <= 0:
                #             kf.handle_bounce()
                #         pure_prediction_x.append(kf.x[0])
                #         pure_prediction_y.append(kf.x[1])
                #         pure_prediction_z.append(kf.x[2])

                #     # Calculate hitting position
                #     x = float(kf.x[0][0])
                #     z = float(kf.x[2][0] - 50)
                #     print('Final prediction: x =', x, 'z =', z)

                #     # Transform to robot coordinates
                #     if x < 1525 // 2:
                #         x = ((1525 / 2) - x) + 70
                #     else:
                #         x = ((1525 / 2) - x) - 70

                #     z = max(40, z)
                #     print('Robot target: x =', x, 'z =', z)

                #     # Execute appropriate hitting motion
                #     if 0 <= x <= 620 and 40 <= z <= 300:
                #         print("Executing forehand hit")
                #         robot.fore_hand(x, z)
                #     elif -620 <= x < 0 and 40 <= z <= 300:
                #         print("Executing backhand hit")
                #         robot.back_hand(x, z)
                #     else:
                #         print("Target out of range, not hitting")

                #     break

                print(f"3D Position: {X.flatten()}")
                num_data += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        print("Cleaning up resources...")
        
        # Clean up robot
        # robot.close()
        
        # Clean up camera resources
        cleanup_resources(left_camera, right_camera, left_tracker, right_tracker, stop_event)
        
        # Plot trajectories if we have enough data
        # if position_3D and (prediction_x or pure_prediction_x):
        #     print("Plotting trajectories...")
        #     plot_trajectories(
        #         position_3D, prediction_x, prediction_y, prediction_z,
        #         pure_prediction_x, pure_prediction_y, pure_prediction_z
        #     )
        
        # Optionally save data
        # if position_3D:
        #     position_3D = np.array(position_3D)
        #     np.savez("data/basler_tracking_data.npz", position_3D=position_3D)
        #     print("Data saved to data/basler_tracking_data.npz")

if __name__ == "__main__":
    main() 