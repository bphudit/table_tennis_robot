### Updated `kalmanEKF.py` (position-only measurements)

import numpy as np

class KalmanFilter:
    def __init__(
        self,
        process_noise_cov,
        measurement_noise_cov,
        initial_state,
        initial_covariance,
        restitution_coeff,
        drag_coeff
    ):
        self.restitution_coeff = restitution_coeff
        self.drag_coeff = drag_coeff
        self.g = -9810  # gravitational acceleration (mm/s^2)

        # State vector: [px, py, pz, vx, vy, vz]ᵀ
        self.x = np.array(initial_state, dtype=float)  # shape (6, 1)

        # Measurement: only positions (px, py, pz)
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))  # 3×6

        self.Q = process_noise_cov            # 6×6 process noise covariance
        self.R = measurement_noise_cov        # 3×3 measurement noise covariance
        self.P = initial_covariance            # 6×6 state covariance

    def predict(self, dt: float):
        # State transition:
        A = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1 - self.drag_coeff*dt, 0, 0],
            [0, 0, 0, 0, 1 - self.drag_coeff*dt, 0],
            [0, 0, 0, 0,  0, 1 - self.drag_coeff*dt]
        ])

        # Control input for gravity on pz and vz
        B = np.array([[0], [0], [0.5 * (dt**2)], [0], [0], [dt]])  # 6×1
        u = np.array([self.g])  # scalar gravity

        # Predict state and covariance
        self.x = A @ self.x + B * u
        self.P = A @ self.P @ A.T + self.Q

    def update(self, z: np.ndarray):
        # z: measurement vector shape (3, 1) = [px, py, pz]ᵀ
        y = z - self.H @ self.x  # residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def handle_bounce(self):
        # Reverse vertical velocity with energy loss, reset pz to zero
        self.x[5] = -self.restitution_coeff * self.x[5]
        self.x[2] = 0


### Updated main script (`kalman_test_velocity.py`)

import sys, os
import numpy as np
from matplotlib import pyplot as plt

# adjust import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.kalmanEKF import KalmanFilter

# load data
data = np.load('prediction_test/data/data_paddle_3.npz')
# extract [x, y, z, timestamp]
position_data = np.array([[p[1], p[0], -p[2], p[3]] for p in data['position_3D']])

# separate positions and timestamps
timestamps = position_data[:, 3]
measurements = position_data[:, :3]  # shape (N, 3)

# initialize Kalman Filter
def diag(vals): return np.diag(vals)

initial_state = [
    [measurements[0, 0]],  # px
    [measurements[0, 1]],  # py
    [measurements[0, 2]],  # pz
    [0],                   # vx unknown → start at zero
    [0],                   # vy
    [0]                    # vz
]
initial_cov = diag([5, 5, 5, 1, 1, 1])
Q = diag([1, 1, 1, 0.5, 0.5, 0.5])
R = diag([0.5, 0.5, 0.5])

kf = KalmanFilter(
    process_noise_cov=Q,
    measurement_noise_cov=R,
    initial_state=initial_state,
    initial_covariance=initial_cov,
    restitution_coeff=0.8,
    drag_coeff=0.5
)

# storage for plotting
pred_x, pred_y, pred_z = [], [], []
pure_x, pure_y, pure_z = [], [], []

# run through measurements
for i in range(len(measurements) - 1):
    # compute dt from timestamps (ns → s)
    dt = (timestamps[i+1] - timestamps[i]) * 1e-9

    # handle bounce if needed
    if kf.x[2] <= 0:
        kf.handle_bounce()
    
    if kf.x[1] <= 2740//2:
        # predict and then update with position-only measurement
        kf.predict(dt)
        z = measurements[i+1].reshape(3, 1)
        kf.update(z)

        pred_x.append(float(kf.x[0]))
        pred_y.append(float(kf.x[1]))
        pred_z.append(float(kf.x[2]))

# pure prediction forward until y reaches table end
dt_fixed = 0.01
while kf.x[1] < 2740:
    kf.predict(dt_fixed)
    if kf.x[2] <= 0:
        kf.handle_bounce()
    pure_x.append(float(kf.x[0]))
    pure_y.append(float(kf.x[1]))
    pure_z.append(float(kf.x[2]))

# 3D plot
ax = plt.axes(projection='3d')
ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')
ax.scatter(
    measurements[:, 0], measurements[:, 1], measurements[:, 2],
    color='r', label='Measured positions'
)
ax.scatter(pred_x, pred_y, pred_z, color='b', label='KF updated')
ax.scatter(pure_x, pure_y, pure_z, color='g', label='KF pure predict')
ax.legend()
plt.show()
