import numpy as np
from scipy.io import loadmat

# Load data
data = loadmat('src/stereoParameters.mat')
K1 = data['K1']
K2 = data['K2']
R_cam = np.linalg.inv(data['R'])
T_cam = -R_cam @ data['T'].reshape((3, 1))

data_left = np.load('src/table_pose/left.npz')
data_right = np.load('src/table_pose/right.npz')

left = data_left['right_table_point']
right = data_right['right_table_point']

# Define points for left and right images
# left = [[-0.51792981,  0.34312107,  1.        ], [-0.63047211, -0.00304284,  1.        ], [ 0.20435657, -0.15932072,  1.        ], [ 0.62597026, -0.06857872,  1.        ]]
# right = [[-0.66238709, -0.06521791,  1.        ], [-0.22565579, -0.18788764,  1.        ], [ 0.65116629, -0.06521791,  1.        ], [0.61085264, 0.33135821, 1.        ]]
result = []

# Calculate 3D coordinates for each point
for i in range(len(left)):
    x1, x2 = np.array(left[i]), np.array(right[i])
    A = np.array([[x1[0], -R_cam[0, 0] * x2[0] - R_cam[0, 1] * x2[1] - R_cam[0, 2]],
                  [x1[1], -R_cam[1, 0] * x2[0] - R_cam[1, 1] * x2[1] - R_cam[1, 2]],
                  [1, -R_cam[2, 0] * x2[0] - R_cam[2, 1] * x2[1] - R_cam[2, 2]]])
    B = T_cam.flatten()
    depths = np.linalg.lstsq(A, B, rcond=None)[0]
    d1 = depths[0]
    X = d1 * x1
    result.append(X)


# Calculate basis vectors and ensure orthogonality
vx, vy = result[1] - result[0], result[3] - result[0]
ux = vx / np.linalg.norm(vx)
uy = vy / np.linalg.norm(vy)

# Calculate uz as the cross product of ux and uy
uz = np.cross(ux, uy)

# Construct the rotation matrix
R = np.array([ux, uy, uz]).T

# Verify orthogonality
print("Orthogonality check:", np.allclose(R @ R.T, np.eye(3), atol=1e-6))

# Translation vector to align top-left as origin
T = np.array(result[0]).reshape((3, 1))

np.savez("pose_parameters.npz", rotation_matrix=R, translation_matrix=T)

transformed_points = []
for point in result:
    point_cam = np.array(point).reshape((3, 1))
    transformed_point = np.round(R.T @ (point_cam - T))/10
    transformed_points.append(transformed_point)
    print("Transformed point:", transformed_point.flatten())








# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# # Assuming you have already computed R, T, and K1 from previous steps

# # Define 3D axes in the table's coordinate system
# origin = np.array([0, 0, 0]).reshape(3, 1)
# x_axis = np.array([100, 0, 0]).reshape(3, 1)  # 100 units along the X-axis
# y_axis = np.array([0, 100, 0]).reshape(3, 1)  # 100 units along the Y-axis
# z_axis = np.array([0, 0, 100]).reshape(3, 1)  # 100 units along the Z-axis

# # Apply rotation and translation to transform table coordinates to camera coordinates
# def transform_to_camera(point, R, T):
#     return R @ point + T

# origin_cam = transform_to_camera(origin, R, T)
# x_axis_cam = transform_to_camera(x_axis, R, T)
# y_axis_cam = transform_to_camera(y_axis, R, T)
# z_axis_cam = transform_to_camera(z_axis, R, T)

# # Project 3D points onto the 2D image plane
# def project_to_image(point, K):
#     point_homog = np.vstack((point, 1))  # Convert to homogeneous coordinates
#     image_point = K @ point_homog[:3]
#     return (image_point[:2] / image_point[2]).flatten()  # Normalize to get (x, y)

# origin_img = project_to_image(origin_cam, K1)
# x_axis_img = project_to_image(x_axis_cam, K1)
# y_axis_img = project_to_image(y_axis_cam, K1)
# z_axis_img = project_to_image(z_axis_cam, K1)

# # Load an example image (left camera image) and plot the axes
# image = cv2.imread("left\img0.png")  # Replace with your actual image file
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# # Plot origin and axes
# plt.scatter(*origin_img, color='cyan', label='Origin')
# plt.plot([origin_img[0], x_axis_img[0]], [origin_img[1], x_axis_img[1]], 'r', label='X-axis')
# plt.plot([origin_img[0], y_axis_img[0]], [origin_img[1], y_axis_img[1]], 'g', label='Y-axis')
# plt.plot([origin_img[0], z_axis_img[0]], [origin_img[1], z_axis_img[1]], 'b', label='Z-axis')

# # Display the plot with a legend
# plt.legend()
# plt.show()



