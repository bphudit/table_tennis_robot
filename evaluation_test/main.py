import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryEvaluator:
    def __init__(self, T, X, Y, Z):
        self.T = T.reshape(-1, 1)  # Ensure T is a column vector
        self.X = X
        self.Y = Y
        self.Z = Z

    def fit_ransac_xy(self):
        # RANSAC fitting for X(T) and Y(T)
        self.ransac_x = RANSACRegressor().fit(self.T, self.X)
        self.ransac_y = RANSACRegressor().fit(self.T, self.Y)

        # Predict fitted x and y values
        self.x_fit = self.ransac_x.predict(self.T)
        self.y_fit = self.ransac_y.predict(self.T)

    def compute_cumulative_s(self, x_points, y_points):
        # Compute cumulative distance along the 2D path
        s = np.insert(np.cumsum(np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2)), 0, 0)
        return s

    def fit_z_along_s(self):
        # Compute path distance 's' from RANSAC XY
        self.s = self.compute_cumulative_s(self.x_fit, self.y_fit)
        # Fit a quadratic curve Z(s)
        self.z_coeff = np.polyfit(self.s, self.Z, 2)

    def generate_fitted_curve(self, num_points=300):
        # Generate smooth T values
        t_dense = np.linspace(self.T.min(), self.T.max(), num_points).reshape(-1, 1)
        x_dense = self.ransac_x.predict(t_dense)
        y_dense = self.ransac_y.predict(t_dense)

        # Compute cumulative s for smooth curve
        s_dense = self.compute_cumulative_s(x_dense, y_dense)
        z_dense = np.polyval(self.z_coeff, s_dense)

        return x_dense, y_dense, z_dense

    def plot_trajectory(self):
        # Generate smooth fitted curve
        x_dense, y_dense, z_dense = self.generate_fitted_curve()

        # Plot raw data and fitted trajectory
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X, self.Y, self.Z, c='blue', label='Raw Data', alpha=0.6)
        ax.plot(x_dense, y_dense, z_dense, c='red', label='Fitted Trajectory (Curved)', linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def run(self):
        self.fit_ransac_xy()
        self.fit_z_along_s()
        self.plot_trajectory()


data = np.load('/Users/bb/Downloads/data1.npz')['position_3D']
evaluator = TrajectoryEvaluator(data[:,3], data[:,0], data[:,1], data[:,2])
evaluator.run()
