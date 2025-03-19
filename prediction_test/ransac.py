
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import time


def ransac(X, y):
    
    # Define RANSAC regressor
    ransac_model = RANSACRegressor(
        LinearRegression(),
        max_trials=5,
        min_samples=2,
        residual_threshold=100,
        random_state=0,
    )

    quadratic = PolynomialFeatures(degree=1)

    
    X_quad = quadratic.fit_transform(X)
    ransac_model = ransac_model.fit(X_quad, y)

    # Get fitted RANSAC curve
    X_fit = np.arange(X.min()-100, X.max()+100, 1)[:, np.newaxis]
    y_quad_fit = ransac_model.predict(quadratic.fit_transform(X_fit))
    

    return ransac_model, X_fit, y_quad_fit

def y_predict_x(ransac_model, y_val):
    # Extract slope and intercept from the underlying LinearRegression model
    base_model = ransac_model.estimator_
    slope = base_model.coef_[0][1]  # Coefficient of x
    intercept = base_model.intercept_[0]  # Intercept

    # Predict x when Y = 0
    y_val
    x_at_y = (y_val-intercept) / slope

    return x_at_y

if __name__ == '__main__':
    data = np.load('data/data3.npz')

    position_3d = data['position_3D']

    X1 = position_3d[:,0].reshape((-1,1))
    y1 = position_3d[:,1].reshape((-1,1))
    a = time.time()
    ransac_model, X_fit, y_quad_fit = ransac(X1, y1)

    x = y_predict_x(ransac_model,0)
    b = time.time()
    print(b-a)
    print(x)

    plt.scatter(X1, y1, c='blue', marker='s', label='Data')
    # Plot fitted RANSAC curve
    plt.plot(
        X_fit,
        y_quad_fit,
        # label="quadratic (d=2), $R^2=%.2f$" % quadratic_r2,
        label='sdfsfe',
        color="red",
        lw=2,
        linestyle="-",
    )

    plt.show()



# Get R2 value
# quadratic_r2 = r2_score(y, ransac.predict(X_quad))

# # Plot inliers
# inlier_mask = ransac.inlier_mask_
# plt.scatter(X[inlier_mask], y[inlier_mask], c="blue", marker="o", label="Inliers")

# # Plot outliers
# outlier_mask = np.logical_not(inlier_mask)
# plt.scatter(
#     X[outlier_mask], y[outlier_mask], c="lightgreen", marker="s", label="Outliers"
# )

# # Plot fitted RANSAC curve
# plt.plot(
#     X_fit,
#     y_quad_fit,
#     label="quadratic (d=2), $R^2=%.2f$" % quadratic_r2,
#     color="red",
#     lw=2,
#     linestyle="-",
# )

# plt.xlabel("X")
# plt.ylabel("Sea ice extent")
# plt.legend(loc="upper left")
# plt.tight_layout()
# plt.show()