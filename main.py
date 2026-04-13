# Find plane of best fit for some points
# How to generate the points?
# Maybe pick a plane and then boost them by some sphere radius

import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import scipy.linalg as LA

def gen_plane():
    class plane: _ = 0
    plane.radius = 3
    plane.pitch_range = [-90, 90] 
    plane.yaw_range = [0, 360]
    n = 100

    plane.pitch = 180 * random.random() - 90
    plane.yaw = 360 * random.random()

    # Define Euler angles (in degrees) for X, Y, Z axes
    euler_angles = [plane.pitch, 0, plane.yaw] # Rotate 90 around X, then 0 Y, then 45 Z

    # Create rotation object (intrinsic rotation)
    plane.pitch_rot = R.from_euler('x', plane.pitch, degrees=True)
    plane.yaw_rot = R.from_euler('z', plane.yaw, degrees=True)
    plane.normal = plane.yaw_rot.apply(plane.pitch_rot.apply(np.array([0, 0, 1])))
    plane.random_r = 0

    plane.points_x = np.zeros(n)
    plane.points_y = np.zeros(n)
    plane.points_z = np.zeros(n)

    for i in range(n):
        # generate point in plane
        p = 2 * (np.array([random.random(), random.random(), 0]) - np.array([0.5, 0.5, 0])) * plane.radius

        # apply rotation
        p = plane.yaw_rot.apply(plane.pitch_rot.apply(p))

        p += plane.random_r * 2 * (np.array([random.random(), random.random(), random.random()]) - np.ones(3) * 0.5)
        plane.points_x[i] = p[0]
        plane.points_y[i] = p[1]
        plane.points_z[i] = p[2]

    return plane

def one_variable_regression(X, Y):
    """
    Given a set of points, find the slope
    """
    
    x_avg = sum(X) / len(X)
    y_avg = sum(Y) / len(Y)

    tilt_axis = 0
    primary_axis = 0
    for i in range(len(X)):
        tilt_axis += (x_avg - X[i]) * (y_avg - Y[i])
        primary_axis += (x_avg - X[i])**2
    return tilt_axis / primary_axis

def normalize(V):
    return V / LA.norm(V)

def regress_2d(plane):
    m_xz = one_variable_regression(plane.points_x, plane.points_z)
    m_yz = one_variable_regression(plane.points_y, plane.points_z)
    # print(m_xz, m_yz)
    forward = normalize(np.array([1, 0, m_xz]))
    right = normalize(np.array([0, 1, m_yz]))
    up = np.cross(forward, right)
    # print(mz_in_axis3D(plane.normal, np.array([1, 0, 0])), mz_in_axis3D(plane.normal, np.array([0, 1, 0])))
    # print(xs.size)
    return up


def mz_in_axis3D(N, axis):
    step = np.dot(N, axis)
    norm = np.array([step, 1])
    recip = np.array([-norm[1], norm[0]])
    #if recip[0] < 0:
    #    recip = - recip
    return recip[1] / recip[0]
    

def z(x, y, v_n):
    return - (v_n[0] * x + v_n[1] * y) / v_n[2]


def graph(plane, up):
    xs = np.arange(-plane.radius, plane.radius, 0.1)
    X, Y = np.meshgrid(xs, xs)
    Z = z(X, Y, plane.normal)
    Z2 = z(X, Y, up)
    fig = go.Figure(data=[go.Scatter3d(x=plane.points_x, y=plane.points_y, z=plane.points_z,
                                       mode='markers'),
                          go.Surface(z=Z, x=X, y=Y ),
                          go.Surface(z=Z2, x=X, y=Y )])
    P0 = np.array([0, 0, 0])
    D = plane.normal

    fig.add_trace(go.Scatter3d(
        x=[P0[0], D[0]],
        y=[P0[1], D[1]],
        z=[P0[2], D[2]],
        mode='lines',
        line=dict(color='blue', width=5),
        name='Vector'
    ))

    fig.show()

def test1():
    plane = gen_plane()
    up = regress_2d(plane)
    
    graph(plane, up)

def analyze():
    norm_weird = np.array([1, 1, 1])
    norm_good = np.array([1, 1, 1])
    n = 0
    for i in range(1000):
        plane = gen_plane()
        up = regress_2d(plane)
        if np.dot(plane.normal, up) < 0.8:
            norm_weird = norm_weird * (n / (n + 1)) + plane.normal * (1 / (n + 1))
            # 1 2 3
            # 0 * 0 + 1 * 1 = 1
            # 1 * 1/2 + 2 * 1/2 = 1.5
            # 1.5 * 2/3 + 3 * 1/3 = 2
        else:
            norm_good = norm_good * (n / (n + 1)) + plane.normal * (1 / (n + 1))
    print(norm_weird, norm_good)
    # result:
    # our regression seem to fail for planes with normals in the xy plane

# analyze()
test1()
