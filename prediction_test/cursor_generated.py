import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kalmanEKF import KalmanFilter 
import numpy as np
from matplotlib import pyplot as plt
import time
from plot_3D import plot_3d

data = np.load('prediction_test/data/data_paddle_1.npz')

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


# initial_state = [[measurements[0][0]], 
#                  [measurements[0][1]],
#                  [measurements[0][2]],
#                  [measurements[0][3]],
#                  [measurements[0][4]],
#                  [measurements[0][5]]]

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
    [0,    0,    0,    1,    0,    0],   # Variance for Vx
    [0,    0,    0,    0,    1,    0],   # Variance for Vy
    [0,    0,    0,    0,    0,    100] # Variance for Vz (more uncertainty)
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
                  restitution_coeff=0.85, 
                  drag_coeff=0.3)


prediction_x = []
prediction_y = []
prediction_z = []

pure_predict_x = []
pure_predict_y = []
pure_predict_z = []

# Error monitoring arrays
innovation_errors = []  # Measurement innovation (y = z - H*x)
innovation_mahalanobis = []  # Normalized innovation (Mahalanobis distance)
covariance_traces = []  # Trace of the error covariance matrix
position_errors = []  # Euclidean distance between prediction and measurement
velocity_magnitudes = []  # Magnitude of velocity vector from KF
iteration_numbers = []  # To track which iteration we're on


for i in range(len(measurements) - 1):
    if kf.x[2] <= 0:  
        kf.handle_bounce()

    if position_3D[i][1] <= 2740:
        # Store pre-update state for calculating error
        x_pred = kf.x.copy()
        
        # Predict the next state
        kf.predict(dt_ls[i])
        print(f"{kf.x} {i} Prediction")
        
        # Store covariance trace before update
        covariance_traces.append(np.trace(kf.P))
        
        # Current measurement as column vector
        z = np.array([[measurements[i+1][0]], 
                      [measurements[i+1][1]],
                      [measurements[i+1][2]]])
        
        # Calculate innovation (pre-update)
        # We need to use the H matrix to project state to measurement space
        # Since we only have position measurements, we use the first 3 rows of H matrix
        H_pos = kf.H[:3, :]  # Take only position rows from H matrix
        innovation = z - H_pos @ kf.x
        innovation_errors.append(np.linalg.norm(innovation))
        
        # Calculate Mahalanobis distance (normalized innovation)
        S = H_pos @ kf.P @ H_pos.T + kf.R
        mahalanobis = float(innovation.T @ np.linalg.inv(S) @ innovation)
        innovation_mahalanobis.append(mahalanobis)
        
        # Calculate position error
        pos_error = np.linalg.norm(z - kf.x[:3])
        position_errors.append(pos_error)
        
        # Calculate velocity magnitude (not error, since we have no velocity measurements)
        vel_magnitude = np.linalg.norm(kf.x[3:6])
        velocity_magnitudes.append(vel_magnitude)
        
        # Store iteration number
        iteration_numbers.append(i)
        
        # Update the filter with measurement
        kf.update(z)
        print(f"{z} {i} Actual (Measurement)")
        
        prediction_x.append(kf.x[0])
        prediction_y.append(kf.x[1])
        prediction_z.append(kf.x[2])
        print(f"{kf.x} {i} Updated")
        print("--------------------------------")

dt = 0.01
while kf.x[1] < 2740:
    kf.predict(dt)

    if kf.x[2] <= 0:  
        kf.handle_bounce()

    print(f"{kf.x} {i} Predict ")
    
    pure_predict_x.append(kf.x[0])
    pure_predict_y.append(kf.x[1])
    pure_predict_z.append(kf.x[2])


# Create 3D trajectory plot
plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')

ax.set_xlim([1525 , 0])
ax.set_ylim([0, 2740])
ax.set_zlim([0, 1500])

ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_zlabel('z [mm]')

ax.scatter(position_3D[:,0], position_3D[:,1], position_3D[:,2], marker = 'o', alpha=1, color='r', label='Coordinate from triangulation')
ax.scatter(prediction_x, prediction_y, prediction_z, marker = 'o', alpha=1, color='b', label='KF prediction and update')
ax.scatter(pure_predict_x, pure_predict_y, pure_predict_z, marker = 'o', alpha=1, color='g', label='KF pure prediction')

ax.legend()
plt.title('3D Trajectory')
plt.show()

# Now plot the error metrics
plt.figure(figsize=(18, 10))

# Innovation error
plt.subplot(2, 2, 1)
plt.plot(iteration_numbers, innovation_errors, marker='o')
plt.title('Innovation Magnitude')
plt.xlabel('Iteration')
plt.ylabel('Error Magnitude')
plt.grid(True)

# Mahalanobis distance
plt.subplot(2, 2, 2)
plt.plot(iteration_numbers, innovation_mahalanobis, marker='o', color='g')
plt.title('Mahalanobis Distance (Normalized Innovation)')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.grid(True)

# Covariance trace
plt.subplot(2, 2, 3)
plt.plot(iteration_numbers, covariance_traces, marker='o', color='r')
plt.title('Trace of Error Covariance Matrix')
plt.xlabel('Iteration')
plt.ylabel('Trace Value')
plt.grid(True)

# Position error and velocity magnitude
plt.subplot(2, 2, 4)
plt.plot(iteration_numbers, position_errors, marker='o', label='Position Error')
plt.plot(iteration_numbers, velocity_magnitudes, marker='x', label='Velocity Magnitude')
plt.title('Position Error and Velocity Magnitude')
plt.xlabel('Iteration')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

