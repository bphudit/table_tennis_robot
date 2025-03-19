import numpy as np

class KalmanFilter:
    def __init__(self, process_noise_std, measurement_noise_std, initial_state, initial_covariance, restitution_coeff):
        self.restitution_coeff = restitution_coeff
        self.g = -9.81  
        
        self.x = np.array(initial_state)
        
        self.H = np.array([[1, 0],
                           [0, 1]])
        
        # Process noise covariance matrix
        self.Q = np.array([[1, 0],
                           [0 ,1]])
        
        # Measurement Noise
        self.R = np.array([[2, 0],
                           [0, 2]])
        # Initial covariance matrix (Higher Value = Adapt Quickly But might over-shoot)
        self.P = initial_covariance
    
    def predict(self, dt):
        
        self.A = np.array([[1, dt], 
                           [0, 1]])
        
        
        self.B = np.array([[0.5 * (dt**2)], 
                           [dt]])
        
        
        u = np.array([self.g])

        # Predict 
        self.x = self.A @ self.x + self.B * u 
        
        # Predict error covariance
        self.P = self.A @ self.P @ self.A.T + self.Q 
        
    
    def update(self, z):
        # Measurement update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)


        # Update state
        self.x = self.x + K @ y
        # Update the error covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def handle_bounce(self):
        self.x[1] = -self.restitution_coeff * self.x[1]
        self.x[0] = 0 
















