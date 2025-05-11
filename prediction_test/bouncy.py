import sys
import os
from trajectory_evaluator import *

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kalmanEKF import KalmanFilter 
import numpy as np
from matplotlib import pyplot as plt
import time
from plot_3D import plot_3d

data = np.load('/Users/bb/Downloads/data_bounce/data_right11.npz')

data_file = '/Users/bb/Downloads/data_bounce/data_right11.npz'
TABLE_BOUNDARIES = {
        'x_min': 0,
        'x_max': 1525,
        'y_min': 0,
        'y_max': 2740
    }
    
    # Run analysis with table boundaries
results = analyze_multiple_trajectories(
    data_file,
    min_bounce_height=None,  # Auto-determine
    smoothing_window=3,
    time_gap_threshold=100,
    table_boundaries=TABLE_BOUNDARIES,
    show_table_outline=False  # Just show table outline, not the green surface
)


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



# kf = KalmanFilter(process_noise_std=Q,
#                   measurement_noise_std=R,
#                   initial_state=initial_state, 
#                   initial_covariance=initial_cov, 
#                   restitution_coeff=0.9, 
#                   drag_coeff=0.3)


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

p_ls = []

def predict_and_update(kf, prediction_x, prediction_y, prediction_z, bounced, count, num_bounced, iteration_numbers, num_data, index):
    
    for i in range(num_data):
    # for i in range(len(measurements) - 1):
        # print('--------------------------------')
        # print(kf.P)
        # print('--------------------------------')


        # if position_3D[i][1] <= 2740//2:
        p_ls.append(kf.P)
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

def pure_predict(kf, dt_ls, bounced, count, num_bounced, iteration_numbers2, qqqqqqpure_predict_x, pure_predict_y, pure_predict_z):
    dt = 0.01
    # while kf.x[1] < 2740:
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

    return i 


rmse_x_ls = []
rmse_y_ls = []
rmse_z_ls = []

for i in range(98):
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

    predict_and_update(kf, prediction_x, prediction_y, prediction_z, bounced, count, num_bounced, iteration_numbers, num_data=i, index=index)
    a = pure_predict(kf, dt_ls, bounced, count, num_bounced, iteration_numbers2, pure_predict_x, pure_predict_y, pure_predict_z)



    rmse_x = np.sqrt(np.mean((np.array(pure_predict_x) - measurements[iteration_numbers2[0]:a,0])**2))
    rmse_y = np.sqrt(np.mean((np.array(pure_predict_y) - measurements[iteration_numbers2[0]:a,1])**2))
    rmse_z = np.sqrt(np.mean((np.array(pure_predict_z) - measurements[iteration_numbers2[0]:a,2])**2))

    rmse_x_ls.append(rmse_x)
    rmse_y_ls.append(rmse_y)
    rmse_z_ls.append(rmse_z)

    print(i)
    print(f"RMSE X: {rmse_x:.2f} mm")
    print(f"RMSE Y: {rmse_y:.2f} mm") 
    print(f"RMSE Z: {rmse_z:.2f} mm")
    print('--------------------------------')

    # Add 3D plot as the fourth subplot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([0,1525])
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
plt.plot(rmse_x_ls, label='RMSE X')
plt.plot(rmse_y_ls, label='RMSE Y')
plt.plot(rmse_z_ls, label='RMSE Z')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('RMSE [mm]')
plt.title('RMSE of X, Y, Z')

# Add 3D plot as the fourth subplot
ax = plt.subplot(2, 2, 2, projection='3d')
ax.set_xlim([1525 , 0])
ax.set_ylim([0, 2740])
ax.set_zlim([0, 1500])

ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_zlabel('z [mm]')

ax.scatter(position_3D[:,0], position_3D[:,1], position_3D[:,2], marker = 'o', alpha=1, color='r', label='Coordinate from triangulation')
# ax.scatter(prediction_x, prediction_y, prediction_z, marker = 'o', alpha=1, color='b', label='KF prediction and update')
# ax.scatter(pure_predict_x, pure_predict_y, pure_predict_z, marker = 'o', alpha=1, color='orange', label='KF pure prediction')
ax.set_title('3D Trajectory')
ax.legend()

plt.tight_layout()
plt.show()






