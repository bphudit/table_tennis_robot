import numpy as np
import copy

class KalmanFilter:
    def __init__(self, process_noise_std, measurement_noise_std, initial_state, initial_covariance, restitution_coeff, drag_coeff):
        self.restitution_coeff = restitution_coeff
        self.drag_coeff = drag_coeff  # new drag coefficient parameter
        self.g = -9810  # gravitational acceleration
        
        #State vector
        self.x = np.array(initial_state)
        #State to measurement matrix [(3,6) cause we only have position measurements]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ]).reshape(3, 6)
        
        #Process noise matrix
        self.Q = process_noise_std
        #Measurement noise matrix
        self.R = measurement_noise_std
        #Initial covariance matrix
        self.P = initial_covariance

    def predict(self, dt):
        #State transition matrix
        self.A = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1 - self.drag_coeff * dt,  0,  0],
            [0, 0, 0, 0,  1 - self.drag_coeff * dt ,  0],
            [0, 0, 0, 0,  0,   1 - self.drag_coeff * dt]
        ])
        #Control input matrix
        self.B = np.array([
            [0],
            [0],
            [0.5 * (dt**2)],
            [0],
            [0],
            [dt]
        ])
        #Control input vector
        u = np.array([self.g])
        # State prediction: positions and velocities are updated.
        self.x = self.A @ self.x + (self.B * u)
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
        self.x[2] = 20

def init_KF(initial_state):
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
                        restitution_coeff=0.935, 
                        drag_coeff=0.4)
    
    return kf


def main():
    kf = init_KF([[1], [1], [1], [100], [100], [100]])
    import time
    a = time.perf_counter()
    kf2 = copy.deepcopy(kf)
    kf.predict(0.01)
    b = time.perf_counter()
    print(b-a)
    print(kf.x[1])
    print(kf2.x[1])

if __name__ == "__main__":
    main()



























