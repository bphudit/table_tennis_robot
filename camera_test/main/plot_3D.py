from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import pyplot as plt


def plot_3d(x, y, z, prediction_x, prediction_y, prediction_z, pure_prediction_x, pure_prediction_y, pure_prediction_z):
    # Load the STL file
    table_mesh = mesh.Mesh.from_file('src/table_tennis.stl')
    # Extract the faces from the mesh
    faces = table_mesh.vectors
    # Create a collection for the 3D model
    table_collection = Poly3DCollection(faces, alpha=0.1, edgecolor='k')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect([1525, 2740, 1500])

    ax.add_collection3d(table_collection)

    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')

    # Set the range for each axis
    ax.set_xlim([0, 1525])
    ax.set_ylim([0, 2740])
    ax.set_zlim([0, 1500])

    ax.scatter(x, y, z, marker = 'o', alpha=1, color='r', label='Coordinate from triangulation')
    ax.scatter(prediction_x, prediction_y, prediction_z, marker = 'o', alpha=1, color='b', label='KF prediction and update')
    ax.scatter(pure_prediction_x, pure_prediction_y, pure_prediction_z, marker = 'o', alpha=1, color='orange', label='KF pure prediction')


    # ax.plot(xfit, yfit, zs=0, color='b', label='2D line (XY)')

    plt.show()

if __name__ == '__main__':
    data = np.load('data/data3.npz')
    position_3d = data['position_3D']

    plot_3d(position_3d[:,0], position_3d[:,1], position_3d[:,2], 0, 0)