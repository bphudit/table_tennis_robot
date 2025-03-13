import numpy as np
import cv2
import time
from threading import Thread, Event
from queue import Queue
from scipy.io import loadmat
import pyrealsense2 as rs
from matplotlib import pyplot as plt
from src.plot_3D import plot_3d
from src.intel import init_cam_color
from src.kalmanEKF import KalmanFilter
import URController.URBasic as URBasic

def fore_hand(x,z, ur):
    if x >= 0:
        if z < 40:
            z = 40
        else:
            z = z
        target_pose_fore = [x/1000, 300/1000, z/1000, 1.240, 1.168, 1.306]
        ur.movej(pose=target_pose_fore, a=100, v=100)
    else:
        print("X IS LESS THAN 0")

def stand_by(ur):
    target_pose_standby = [91/1000, 625/1000, 418/1000, 0.125, -1.71, -2.639]
    ur.movej(pose=target_pose_standby, a=2, v=2)

def back_hand(x,z, ur):
    if x < 0 :
        if z < 40:
            z = 40
        else:
            z = z
        target_pose_back = [x/1000, 300/1000,z/1000, 1.206, -1.217, -1.187]
        ur.movej(pose=target_pose_back, a=100, v=100)
    else:
        print('X IS MORE THAN 0')

robot = URBasic.robotModel.RobotModel()
robot_ip = '10.10.0.7'


def init_KF(measurements):
    initial_state = measurements

    initial_cov = np.array([[2, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 0, 0],
                            [0, 0, 5, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

    #Process Noise (Lower = trust more in model)
    Q = np.array([
        [0.1, 0,    0,    0,    0,    0],   # Variance for Px
        [0,    0.1, 0,    0,    0,    0],   # Variance for Py
        [0,    0,    2, 0,    0,    0],   # Variance for Pz (more uncertainty)
        [0,    0,    0,    0.5, 0,    0],   # Variance for Vx
        [0,    0,    0,    0,    0.5, 0],   # Variance for Vy
        [0,    0,    0,    0,    0,    5] # Variance for Vz (more uncertainty)
    ])


    #Measurement Noise
    R = np.array([
        [1.0, 0,    0,    0,    0,    0],   # Variance for measured Px
        [0,    1.0, 0,    0,    0,    0],   # Variance for measured Py
        [0,    0,    0.5, 0,    0,    0],   # Variance for measured Pz
        [0,    0,    0,    5.0, 0,    0],   # Variance for measured Vx
        [0,    0,    0,    0,    5.0, 0],   # Variance for measured Vy
        [0,    0,    0,    0,    0,    2.0]# Variance for measured Vz
    ])

    kf = KalmanFilter(process_noise_std=Q,measurement_noise_std=R,initial_state=initial_state, initial_covariance=initial_cov, restitution_coeff=0.8)
    return kf

data = loadmat('stereoParameters.mat')
data2 = np.load('pose_parameters.npz')

pose_rotation = data2['rotation_matrix']
pose_translation = data2['translation_matrix']

# Intrinsic matrices (example values)
K1 = data['K1']

K2 = data['K2']

# Rotation and translation from Camera 2 to Camera 1
R = data['R']
T = data['T'].reshape((3,1))

R = np.linalg.inv(R)
T = -R @ T

K1_inv = np.linalg.inv(K1)
K2_inv = np.linalg.inv(K2)

pipeline_right, pipeline_left = init_cam_color()

# Queue to store frames for display in the main thread
left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)

x_1 = Queue(maxsize=1)
x_2 = Queue(maxsize=1)

lower_limit_left = np.array([0, 212, 141])
lower_limit_right = np.array([0,109,151])
upper_limit = np.array([81,255,255])

stop_event = Event()

# Function to capture and process frames
def capture_and_process(pipeline, intrinsic_inv, output_queue, val_queue, lower_limit):
    try:
        while not stop_event.is_set():
            u , v = None, None
            frame = pipeline.wait_for_frames()
            color_frame = frame.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())


            # Preprocessing
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            blurred_image = cv2.GaussianBlur(hsv_image, (7, 7), 0)
            mask = cv2.inRange(blurred_image, lower_limit, upper_limit)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if 10 <= cv2.contourArea(contour) <= 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    u, v = x + w // 2, y + h // 2
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    break
            
            if u and v:
                if val_queue.full():
                    val_queue.get()
                x = intrinsic_inv @ np.array([u, v, 1])
                val_queue.put(x)

            # Put the processed frame into the output queue for the main thread
            if output_queue.full():
                output_queue.get()  # Remove the previous frame if full
            output_queue.put(color_image)
    except Exception as e:
        pass
# Set up threads
left_thread = Thread(target=capture_and_process, args=(pipeline_left, K1_inv, left_queue, x_1, lower_limit_left))
right_thread = Thread(target=capture_and_process, args=(pipeline_right, K2_inv, right_queue, x_2, lower_limit_right))

left_thread.start()
right_thread.start()

position_x = []
position_y = []
position_z = []
num = 0

X1, y1 = None, None

position_3D = []

num_data = 0
prediction_x = []
prediction_y = []
prediction_z = []

pure_prediction_x = []
pure_prediction_y = []
pure_prediction_z = []


# Main loop to display frames
ur = URBasic.urScriptExt.UrScriptExt(host=robot_ip, robotModel=robot)
stand_by(ur)
while True:
    if not left_queue.empty():
        left_frame = left_queue.get()
        cv2.imshow("Left Camera", left_frame)

    if not right_queue.empty():
        right_frame = right_queue.get()
        cv2.imshow("Right Camera", right_frame)

    if not x_1.empty()and not x_2.empty() :
        x1 = x_1.get()
        x2 = x_2.get()

        # Set up and solve for depths d1 and d2
        A = np.array([[x1[0], -R[0, 0] * x2[0] - R[0, 1] * x2[1] - R[0, 2]],
                      [x1[1], -R[1, 0] * x2[0] - R[1, 1] * x2[1] - R[1, 2]],
                      [1, -R[2, 0] * x2[0] - R[2, 1] * x2[1] - R[2, 2]]])
        B = T.flatten()

        # Solve for d1 and d2
        depths = np.linalg.lstsq(A, B, rcond=None)[0]
        d1 = depths[0]
        d2 = depths[1]

        # Compute the 3D point in Camera 1's frame
        X = d1 * x1
        #Transformation to table coordinate
        X = pose_rotation.T @ (X.reshape((3,1)) - pose_translation)

        t = time.time()

        position_x.append(X[0])
        position_y.append(X[1])
        position_z.append(-X[2] - 40)
        x, y, z = float(X[0]), float(X[1]), float(-X[2]-40)
        position_3D.append([x, y, z, t])

        if y >= 2740//2:
            if num_data == 1:
                dt = position_3D[num_data][-1] - position_3D[num_data-1][-1]
                dt = max(dt, 0.00001)

                velocity_x = (position_3D[num_data][0] - position_3D[num_data-1][0])/dt
                velocity_y = (position_3D[num_data][1] - position_3D[num_data-1][1])/dt
                velocity_z = (position_3D[num_data][2] - position_3D[num_data-1][2])/dt

                initial_val = np.array([[x],
                                        [y],
                                        [z],
                                        [velocity_x],
                                        [velocity_y],
                                        [velocity_z]])

                kf = init_KF(measurements=initial_val)

                kf.predict(dt)
            elif num_data > 1:
                dt = position_3D[num_data][-1] - position_3D[num_data-1][-1]
                dt = max(dt, 0.00001)

                velocity_x = (position_3D[num_data][0] - position_3D[num_data-1][0])/dt
                velocity_y = (position_3D[num_data][1] - position_3D[num_data-1][1])/dt
                velocity_z = (position_3D[num_data][2] - position_3D[num_data-1][2])/dt

                kf.update(z=np.array([[x],
                                    [y],
                                    [z],
                                    [velocity_x],
                                    [velocity_y],
                                    [velocity_z]]))
                
                kf.predict(dt)

                if kf.x[2] <= 0:  
                    kf.handle_bounce()

                prediction_x.append(kf.x[0])
                prediction_y.append(kf.x[1])
                prediction_z.append(kf.x[2])
        else:
            a = time.perf_counter()
            # Stop the threads
            stop_event.set()

            # Wait for threads to finish
            left_thread.join()
            right_thread.join()

            dt = 0.01
            while (kf.x[1] > 0):
                kf.predict(dt)

                if kf.x[2] <= 0:  
                    kf.handle_bounce()

                pure_prediction_x.append(kf.x[0])
                pure_prediction_y.append(kf.x[1])
                pure_prediction_z.append(kf.x[2])

            x = float(kf.x[0][0])
            z = float(kf.x[2][0] - 50)
            print('-------z-------', z)
            print('-------x-------', x)

            if x < 1525//2:
                x = ((1525/2) - x) + 70
            else:
                x = ((1525/2) - x) - 70
        
            print('-------x-------', x)
            print('-------z-------', z)

            if z < 40:
                z = 40
            b = time.perf_counter()

            if 0 <= x <= 620 and 40 <= z <= 300:
                
                fore_hand(x,z, ur)
            elif -620 <= x < 0 and 40 <= z <= 300:
                
                back_hand(x,z, ur)
            
            break

        
        print(X)
        num_data += 1


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

ur.close()


# print(b-a)
pipeline_left.stop()
pipeline_right.stop()
cv2.destroyAllWindows()

position_3D = np.array(position_3D)
ax = plt.axes(projection='3d')

ax.set_xlim([1525 , 0])
ax.set_ylim([0, 2740])
ax.set_zlim([0, 1500])

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('height [m]')

ax.scatter(position_3D[:,0], position_3D[:,1], position_3D[:,2], marker = 'o', alpha=1, color='r', label='Data from Sensor')
ax.scatter(prediction_x, prediction_y, prediction_z, marker = 'o', alpha=1, color='b', label='KF prediction')
ax.scatter(pure_prediction_x, pure_prediction_y, pure_prediction_z, marker = 'o', alpha=1, color='g', label='KF pure prediction')

ax.legend()

plt.show()

# plot_3d(position_x, position_y, position_z, X1, y1)

# print(np.min(position_3D[:,2]))
# np.savez("data/data5.npz", position_3D=position_3D)
