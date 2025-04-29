from kalman import KalmanFilter 
import numpy as np
from matplotlib import pyplot as plt
from plot_3D import plot_3d

data = np.load('prediction_test/data/data_paddle.npz')

position_3D = data['position_3D']

plot_3d(position_3D[:,1], position_3D[:,0], -position_3D[:,2])

# # Sample data (replace with your actual data)
# height = position_3D[:, 0]
# time = position_3D[:, 1] 

# # Create the plot
# plt.scatter(time, height)

# # Add labels and title
# plt.xlabel('Time')
# plt.ylabel('Height')
# plt.title('Height vs Time')

# # Show the plot
# plt.show()




measurements = []

dt_ls = []

#OLD
# for i in range(1, len(y)):
    # dt_ls.append(x[i] - x[i-1])
    # velocity = (y[i] - y[i-1])/(x[i] - x[i-1])
    # measurements.append((y[i], velocity))

#NEW
# for i in range(1, len(position_3D[:,-1])):
#     dt = position_3D[i][-1] - position_3D[i-1][-1]
#     dt_ls.append(dt)
#     velocity = position_3D[i][1]



# initial_state = [[measurements[0][0]], 
#                  [measurements[0][1]]]

# initial_cov = np.array([[1, 0],
#                         [0 ,1]])


# kf = KalmanFilter(0, 0, initial_state, initial_cov, 0.8)


# print(measurements[1:10])

# prediction = []

# for i in range(1, len(x[1:-1])):

#     kf.predict(dt_ls[i])
#     if kf.x[0] <= 0:  
#         kf.handle_bounce()
#         print('HANDLED')
#     # prediction.append(kf.x[0])
#     print(f"{kf.x} {i} Predict ")

#     if horiz[i] > 0:
#         kf.update(z=np.array([[measurements[i+1][0]], 
#                         [measurements[i+1][1]]]))
#         prediction.append(kf.x[0])
#         print(f"{kf.x} {i} Update ")
#     else:
#         prediction.append(kf.x[0])

# error_sum = 0
# for i in range(len(prediction)):
#     error_sum += (prediction[i] - y[3:][i])**2

# rmse = error_sum/len(prediction)**0.5

# print(f"RMSE: {rmse}")

# plt.scatter(x[2:], y[2:], color='red', label='D435 Depth Output')
# plt.scatter(x[3:], prediction, color='blue', label='Kalman Filter')

# # Label 
# plt.xlabel('Time')
# plt.ylabel('Position')

# plt.legend()

# plt.show()
