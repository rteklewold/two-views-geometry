import numpy as np
import cv2
import open3d as o3d


K = np.array([
    [7.215377e2,        0,              6.095593e2],
    [0,                 7.215377e2,     1.728540e2],
    [0,                 0,              1]
])  # intrinsic matrix of camera

Bf = 3.875744e+02  # base line * focal length

# load images
left = cv2.imread('/home/ecn/lab2_avg/images_lab2/left.png', 0)
right = cv2.imread('/home/ecn/lab2_avg/images_lab2/right.png', 0)
left_color = cv2.imread('/home/ecn/lab2_avg/images_lab2//left_color.png')

# compute disparity
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(left, right)  # an image of the same size with "left" (or "right")

#print(disparity)
# TODO: compute depth of every pixel whose disparity is positive
# hint: assume d is the disparity of pixel (u, v)
# hint: the depth Z of this pixel is Z = Bf / d
depth = []
#print(depth.shape)
for i in range(disparity.shape[0]):
    for j in range(disparity.shape[1]):
        if disparity[i][j] > 0:
            depth.append(Bf/disparity[i][j])


# TODO: compute normalized coordinate of every pixel whose disparity is positive
# hint: the normalized coordinate of pixel [u, v, 1] is K^(-1) @ [u, v, 1]
norm_pixels = []
for i in range(disparity.shape[0]):
    for j in range(disparity.shape[1]):
        if disparity[i][j] > 0:
            norm_pixels.append((np.linalg.inv(K)@np.array([i, j, 1])).transpose())


norm_pixels = np.asarray(norm_pixels)
print('norm_pixels shape' , norm_pixels.shape)
# TODO: compute 3D coordinate of every pixel whose disparity is positive
# hint: 3D coordinate of pixel (u, v) is the product of Z and its normalized cooridnate
coord_3d = []
for i in range(len(depth)):
    coord_3d.append(depth[i] * norm_pixels[i, :])

#all_3d = np.array([])  # this is matrix storing 3D coordinate of every pixel whose disparity is positive
# the shape of all_3d is <num_pixels_positive_disparity, 3>
# each row of all_3d is a 3D coordinate [X, Y, Z]
# you need to change the value of all_3d with your computation of 3D coordinate of every pixel whose disparity is positive
all_3d = np.asarray(coord_3d)
print('all_3d shape:', all_3d.shape)

# TODO: get color for 3D points
#all_color = np.array([])  # this is matrix storing color of every pixel whose disparity is positive
# TODO: THE ORDER OF all_color IS THE SAME WITH all_3d
# the shape of all_color is <num_pixels_positive_disparity, 3>
# each row of all_color is [R, G, B] value
all_color = []
for i in range(disparity.shape[0]):
    for j in range(disparity.shape[1]):
        if disparity[i][j] > 0:
            all_color.append(left_color[i][j])

all_color = np.asarray(all_color)
print('all_color shape:', all_color.shape)
# normalize all_color
all_color = all_color.astype(float) / 255.0

all_3d_new = []
all_color_new = []
for i in range(all_3d.shape[0]):
    if all_3d[i][0] < 10 and all_3d[i][0]  > -10 and all_3d[i][1] < 5 and all_3d[i][1]  > -5 and  all_3d[i][2] < 30 and all_3d[i][2] > 5:
        all_3d_new.append(all_3d[i])
        all_color_new.append(all_color[i])
all_3d_new = np.asarray(all_3d_new)
all_color_new = np.asarray(all_color_new)
print(all_3d_new.shape)
# Display pointcloud
cloud = o3d.geometry.PointCloud()  # create pointcloud object
cloud.points = o3d.utility.Vector3dVector(all_3d_new)
cloud.colors = o3d.utility.Vector3dVector(all_color_new)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])  # create frame object
o3d.visualization.draw_geometries([cloud, mesh_frame])
