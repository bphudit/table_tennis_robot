import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kalmanEKF import KalmanFilter 
import numpy as np
from matplotlib import pyplot as plt
import time
from plot_3D import plot_3d

data = np.load('prediction_test/data/data_paddle_4.npz')

position_3D = np.array([[pos[1], pos[0], -pos[2], pos[3]] for pos in data['position_3D']])

measurements = []
dt_ls = []


for i in range(1, len(position_3D[:,-1])):
    dt = (position_3D[i][-1] - position_3D[i-1][-1]) * (10**-9)
    dt_ls.append(dt)
    velocity_x = (position_3D[i][0] - position_3D[i-1][0])/dt
    velocity_y = (position_3D[i][1] - position_3D[i-1][1])/dt
    velocity_z = (position_3D[i][2] - position_3D[i-1][2])/dt
    measurements.append((position_3D[i][0], position_3D[i][1], position_3D[i][2]))


measurements = np.array(measurements)


initial_state = [[measurements[0][0]], 
                 [measurements[0][1]],
                 [measurements[0][2]],
                 [0],
                 [0],
                 [0]]

initial_cov = np.array([[5, 0, 0, 0, 0, 0],
                        [0, 5, 0, 0, 0, 0],
                        [0, 0, 5, 0, 0, 0],
                        [0, 0, 0, 1000, 0, 0],
                        [0, 0, 0, 0, 1000, 0],
                        [0, 0, 0, 0, 0, 1000]])

#Process Noise (Lower = trust more in model)
Q = np.array([
    [1,    0,    0,    0,    0,    0],   # Variance for Px
    [0,    1,    0,    0,    0,    0],   # Variance for Py
    [0,    0,    1,    0,    0,    0],   # Variance for Pz (more uncertainty)
    [0,    0,    0,    1000,    0,    0],   # Variance for Vx
    [0,    0,    0,    0,    1000,    0],   # Variance for Vy
    [0,    0,    0,    0,    0,    1000] # Variance for Vz (more uncertainty)
])


#Measurement Noise
R = np.array([
    [5**2,  0,     0],   # Variance for measured Px
    [0,    5**2,   0],   # Variance for measured Py
    [0,    0,   5**2]    # Variance for measured Pz
])



kf = KalmanFilter(process_noise_std=Q,
                  measurement_noise_std=R,
                  initial_state=initial_state, 
                  initial_covariance=initial_cov, 
                  restitution_coeff=0.9, 
                  drag_coeff=0.5)


prediction_x = []
prediction_y = []
prediction_z = []

pure_predict_x = []
pure_predict_y = []
pure_predict_z = []

index = 0

bounced = False
count = 0
num_bounced = 0

iteration_numbers = []
iteration_numbers2 = []

for i in range(len(measurements) - 1):
    # print('--------------------------------')
    # print(kf.P)
    # print('--------------------------------')


    if position_3D[i][1] <= 2740//2:

        kf.predict(dt_ls[i])
        if bounced:
            count += 1
            if count >= 10:
                count = 0
                bounced = False

        if kf.x[2] <= 20:  
            if not bounced:
                kf.handle_bounce()
                bounced = True
                num_bounced += 1
                # print("Bounce--------------------------------")
        # print(f"{kf.x} {i} Prediction")
        kf.update(z=np.array([[measurements[i+1][0]], 
                            [measurements[i+1][1]],
                            [measurements[i+1][2]]]))
        # print(f"{measurements[i+1].reshape(-1,1)} {i} Actual (Measurement)")
        
        prediction_x.append(kf.x[0])
        prediction_y.append(kf.x[1])
        prediction_z.append(kf.x[2])
        index = i
        iteration_numbers.append(i)

    # print('--------------------------------')
    # print(kf.P[2])
    # print(kf.P[-1])
    # print('--------------------------------')
    # print(f"{kf.x} {i} Updated")
    # print("--------------------------------")


dt = 0.01
for i in range(index+1, len(measurements) - 1):
    if kf.x[1] >= 2740:
        break

    kf.predict(dt_ls[i])
    iteration_numbers2.append(i)
    if bounced:
        count += 1
        if count >= 10:
            count = 0
            bounced = False

    if kf.x[2] <= 20:  
        if not bounced:
            kf.handle_bounce()
            bounced = True
            num_bounced += 1
            # print("Bounce--------------------------------")

    # print(f"{kf.x} {i} Predict ")
    
    pure_predict_x.append(kf.x[0])
    pure_predict_y.append(kf.x[1])
    pure_predict_z.append(kf.x[2])

x_dif = measurements[:i,0][-1] - pure_predict_x[-1] 
y_dif = measurements[:i,1][-1] - pure_predict_y[-1] 
z_dif = measurements[:i,2][-1] - pure_predict_z[-1] 

print(f"X: {x_dif} mm")
print(f"Y: {y_dif} mm")
print(f"Z: {z_dif} mm")


# Add 3D plot as the fourth subplot
ax = plt.subplot(1, 1, 1, projection='3d')
ax.set_xlim([1525 , 0])
ax.set_ylim([0, 2740])
ax.set_zlim([0, 1500])

ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_zlabel('z [mm]')

ax.scatter(position_3D[:,0], position_3D[:,1], position_3D[:,2], marker = 'o', alpha=1, color='r', label='Coordinate from triangulation')
ax.scatter(prediction_x, prediction_y, prediction_z, marker = 'o', alpha=1, color='b', label='KF prediction and update')
ax.scatter(pure_predict_x, pure_predict_y, pure_predict_z, marker = 'o', alpha=1, color='orange', label='KF pure prediction')
ax.set_title('3D Trajectory')
ax.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(iteration_numbers, prediction_x, marker='x', label='KF prediction')
plt.plot(iteration_numbers2, pure_predict_x, marker='x', label='KF pure prediction')
plt.plot(np.arange(1,i), measurements[1:i,0], marker='x', label='Actual measurement')
plt.axvline(x=iteration_numbers[-1], color='r', linestyle='--')
plt.title('X Axis')
plt.xlabel('Iteration')
plt.ylabel('X [mm]')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(iteration_numbers, prediction_y, marker='x', label='KF prediction')
plt.plot(iteration_numbers2, pure_predict_y, marker='x', label='KF pure prediction')
plt.plot(np.arange(1,i), measurements[1:i,1], marker='x', label='Actual measurement')
plt.axvline(x=iteration_numbers[-1], color='r', linestyle='--')
plt.title('Y Axis')
plt.xlabel('Iteration')
plt.ylabel('Y [mm]')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(iteration_numbers, prediction_z, marker='x', label='KF prediction')
plt.plot(iteration_numbers2, pure_predict_z, marker='x', label='KF pure prediction')
plt.plot(np.arange(1,i), measurements[1:i,2], marker='x', label='Actual measurement')
plt.axvline(x=iteration_numbers[-1], color='r', linestyle='--')
plt.title('Z Axis')
plt.xlabel('Iteration')
plt.ylabel('Z [mm]')
plt.legend()
plt.grid(True)

# Add 3D plot as the fourth subplot
ax = plt.subplot(2, 2, 4, projection='3d')
ax.set_xlim([1525 , 0])
ax.set_ylim([0, 2740])
ax.set_zlim([0, 1500])

ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_zlabel('z [mm]')

ax.scatter(position_3D[:,0], position_3D[:,1], position_3D[:,2], marker = 'o', alpha=1, color='r', label='Coordinate from triangulation')
ax.scatter(prediction_x, prediction_y, prediction_z, marker = 'o', alpha=1, color='b', label='KF prediction and update')
ax.scatter(pure_predict_x, pure_predict_y, pure_predict_z, marker = 'o', alpha=1, color='orange', label='KF pure prediction')
ax.set_title('3D Trajectory')
ax.legend()

plt.tight_layout()
plt.show()


