import numpy as np

class KalmanFilter:
    def __init__(self, process_noise_std, measurement_noise_std, initial_state, initial_covariance, restitution_coeff, drag_coeff):
        self.restitution_coeff = restitution_coeff
        self.drag_coeff = drag_coeff  # new drag coefficient parameter
        self.g = -9810  # gravitational acceleration
        
        self.x = np.array(initial_state)
        
        # Using a full 6x6 identity for measurement matrix
        self.H = np.eye(6)
        
        self.Q = process_noise_std
        self.R = measurement_noise_std
        self.P = initial_covariance

    def predict(self, dt):
        # Updated state transition matrix incorporating drag.
        # The upper block integrates positions using velocities,
        # while the lower block applies a decay (drag) on the velocities.
        self.A = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1 - self.drag_coeff*dt, 0, 0],
            [0, 0, 0, 0, 1 - self.drag_coeff*dt, 0],
            [0, 0, 0, 0,  0, 1 - self.drag_coeff*dt]
        ])
        
        # The B matrix remains the same for integrating gravity into z.
        # Note that gravity only affects the z-velocity and position.
        self.B = np.array([
            [0],
            [0],
            [0.5 * (dt**2)],
            [0],
            [0],
            [dt]
        ])
        
        u = np.array([self.g])
        
        # State prediction: positions and velocities are updated.
        self.x = self.A @ self.x + self.B * u
        
        # Predict error covariance
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        # Measurement update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def handle_bounce(self):
        # Reverse the vertical velocity with energy loss, and reset the z-position.
        self.x[5] = -self.restitution_coeff * self.x[5]
        self.x[2] = 0
















