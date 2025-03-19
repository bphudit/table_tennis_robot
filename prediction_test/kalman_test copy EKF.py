from src.kalmanEKF import KalmanFilter 
import numpy as np
from matplotlib import pyplot as plt
import time

data = np.load('data/data1.npz')

position_3D = data['position_3D']


measurements = []

dt_ls = []

# for i in range(1, len(z)):
#     dt_ls.append(t[i] - t[i-1])
#     delta = (t[i] - t[i-1])
#     velocity_x = (x[i] - x[i-1])/delta
#     velocity_y = (y[i] - y[i-1])/delta
#     velocity_z = (z[i] - z[i-1])/delta
#     measurements.append((x[i], y[i], z[i], velocity_x, velocity_y, velocity_z))


for i in range(1, len(position_3D[:,-1])):
    dt = position_3D[i][-1] - position_3D[i-1][-1]
    dt_ls.append(dt)
    velocity_x = (position_3D[i][0] - position_3D[i-1][0])/dt
    velocity_y = (position_3D[i][1] - position_3D[i-1][1])/dt
    velocity_z = (position_3D[i][2] - position_3D[i-1][2])/dt
    measurements.append((position_3D[i][0], position_3D[i][1], position_3D[i][2], velocity_x, velocity_y, velocity_z))


measurements = np.array(measurements)


initial_state = [[measurements[0][0]], 
                 [measurements[0][1]],
                 [measurements[0][2]],
                 [measurements[0][3]],
                 [measurements[0][4]],
                 [measurements[0][5]]]

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



kf = KalmanFilter(process_noise_std=Q,measurement_noise_std=R,initial_state=initial_state, initial_covariance=initial_cov, restitution_coeff=0.8, drag_coeff=0.5)

# print(len(position_3D))
# print(measurements[1:5])
# print(initial_state)

prediction_x = []
prediction_y = []
prediction_z = []

pure_predict_x = []
pure_predict_y = []
pure_predict_z = []

from numba import njit

def process(kf, index):
    kf.predict(dt_ls[index])
    if kf.x[2] <= 0:  
        kf.handle_bounce()
    #     print('HANDLED')

    # print(f"{kf.x} {index} Predict ")

    if position_3D[index][1] >= 2740//2:
        kf.update(z=measurements[index+1].reshape(-1,1))
        

        


for i in range(len(measurements) - 1):
    # process(kf, i)
    # kf.predict(dt_ls[i])
    if kf.x[2] <= 0:  
        kf.handle_bounce()
    #     print('HANDLED')

    # print(f"{kf.x} {i} Predict ")

    if position_3D[i][1] >= 2740//2:
        kf.predict(dt_ls[i])
        kf.update(z=np.array([[measurements[i+1][0]], 
                            [measurements[i+1][1]],
                            [measurements[i+1][2]],
                            [measurements[i+1][3]],
                            [measurements[i+1][4]],
                            [measurements[i+1][5]]]))
        print(np.array([[measurements[i+1][0]], 
                            [measurements[i+1][1]],
                            [measurements[i+1][2]],
                            [measurements[i+1][3]],
                            [measurements[i+1][4]],
                            [measurements[i+1][5]]]))
        prediction_x.append(kf.x[0])
        prediction_y.append(kf.x[1])
        prediction_z.append(kf.x[2])
        print(f"{kf.x} {i} Update ")

print(kf.x[1])


dt = 0.01
while kf.x[1] > 0:
    kf.predict(dt)

    if kf.x[2] <= 0:  
        kf.handle_bounce()
    
    pure_predict_x.append(kf.x[0])
    pure_predict_y.append(kf.x[1])
    pure_predict_z.append(kf.x[2])

print(kf.x[1])
print(len(pure_predict_x))




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

# Show the plot
plt.show()

# print(len(position_3D[len(prediction_z)+2:,-1]), len(pure_predict_z))

# plt.scatter(position_3D[:,-1], position_3D[:,2], color='red', label='Sensor')
# plt.scatter(position_3D[1:len(prediction_z)+1,-1], prediction_z, color='blue', label='KF Update')
# plt.scatter(position_3D[len(prediction_z)+2:,-1], pure_predict_z, color='green', label='KF Update')

# Label 
# plt.xlabel('time')
# plt.ylabel('height')

# plt.legend()

# plt.show()


