import numpy as np
import cv2
from matplotlib import pyplot as plt

def normalize_transformation(points: np.ndarray) -> np.ndarray:
    """
    Compute a similarity transformation matrix that translate the points such that
    their center is at the origin & the avg distance from the origin is sqrt(2)
    :param points: <float: num_points, 2> set of key points on an image
    :return: (sim_trans <float, 3, 3>)
    """
    u, v = np.mean(points, axis=0)
    center = np.array([u, v])  # TODO: find center of the set of points by computing mean of x & y
    dist = np.zeros((points.shape[0], 1))  # TODO: matrix of distance from every point to the origin, shape: <num_points, 1>
    for i in range(points.shape[0]):
        dist[i] = np.linalg.norm(points[i]-center)
    s = np.sqrt(2)/np.mean(dist, axis=0)  # TODO: scale factor the similarity transformation = sqrt(2) / (mean of dist)
    sim_trans = np.array([
        [s,     0,      -s * center[0]],
        [0,     s,      -s * center[1]],
        [0,     0,      1]
    ])
    return sim_trans


def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinate
    :param points: <float: num_points, num_dim>
    :return: <float: num_points, 3>
    """
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)


# read image & put them in grayscale
img1 = cv2.imread('/home/ecn/lab2_avg/images_lab2/chapel00.png', 0)  # queryImage
img2 = cv2.imread('/home/ecn/lab2_avg/images_lab2/chapel01.png', 0)  # trainImage

# detect kpts & compute descriptor
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# match kpts
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# organize key points into matrix, each row is a point
query_kpts = np.array([kp1[m.queryIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>
train_kpts = np.array([kp2[m.trainIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>

# normalize kpts
T_query = normalize_transformation(query_kpts)  # get the similarity transformation for normalizing query kpts
#normalized_query_kpts = np.array([])  # TODO: apply T_query to query_kpts to normalize them
query_kpts = homogenize(query_kpts)
normalized_query_kpts = T_query@query_kpts.transpose()
T_train = normalize_transformation(train_kpts)  # get the similarity transformation for normalizing train kpts
#normalized_train_kpts = np.array([])  # TODO: apply T_train to train_kpts to normalize them
train_kpts = homogenize(train_kpts)
normalized_train_kpts = T_train@train_kpts.transpose()
print(normalized_train_kpts.shape)
# construct homogeneous linear equation to find fundamental matrix
A = np.zeros((8, 9))
# TODO: construct A according to Eq.(3) in lab subject
for i in range(8):
    A[i] = np.array([normalized_train_kpts[0][i]*normalized_query_kpts[0][i],normalized_train_kpts[0][i]*normalized_query_kpts[1][i], normalized_train_kpts[0][i],
                  normalized_train_kpts[1][i]*normalized_query_kpts[0][i], normalized_train_kpts[1][i]*normalized_query_kpts[1][i], normalized_train_kpts[1][i],
                  normalized_query_kpts[0][i], normalized_query_kpts[1][i], 1])
    #np.concatenate((A, a), axis = 0)

print(A.shape)
# TODO: find vector f by solving A f = 0 using SVD
# hint: perform SVD of A using np.linalg.svd to get u, s, vh (vh is the transpose of v)
# hint: f is the last column of v
U, S, VT = np.linalg.svd(A)
f = VT[VT.shape[0]-1, :]
#f = np.array([])  # TODO: find f

# arrange f into 3x3 matrix to get fundamental matrix F
F = f.reshape(3, 3)
print('rank F: ', np.linalg.matrix_rank(F))  # should be = 3

# TODO: force F to have rank 2
# hint: perform SVD of F using np.linalg.svd to get u, s, vh
# hint: set the smallest singular value of F to 0
# hint: reconstruct F from u, new_s, vh
u, s, vh = np.linalg.svd(F)
new_s = np.array([[s[0],0,0],[0, s[1], 0],[0, 0, 0]])
F = np.matmul(np.matmul(u, new_s),vh)
print(np.linalg.matrix_rank(F))
assert np.linalg.matrix_rank(F) == 2, 'Fundamental matrix must have rank 2'

# TODO: de-normlaize F
# hint: last line of Algorithme 1 in the lab subject
F = T_train.transpose()@F@T_query
F_gt = np.loadtxt('chapel.00.01.F')
print('The difference between the calculated fundamental matrix and the ground truth is: e=', F - F_gt)
print('F', F)
print('F_gt', F_gt)


# practical 1.1.1
plt.imshow(img1, cmap='gray')
print('Please click on one point on the image:')
x = plt.ginput(1)
x = np.asarray(x)
l_prime = F_gt@np.array([x[0][0], x[0][1], 1])

m, n = img2.shape[:2]
a = np.linspace(0, n, 500)  #x coordinates of points that belong to the epipolar line
b = (l_prime[0]*a + l_prime[2]) / -l_prime[1]   # y coordinates of points that belong to the epipolar line
plt.subplot(1, 2, 1)
plt.plot(x[0][0], x[0][1], marker='o', markersize=10, markerfacecolor='green')
plt.imshow(img1, cmap='gray')

plt.subplot(1, 2, 2)
ndx = (b>0) & (b<m)
plt.plot(a[ndx], b[ndx])
plt.imshow(img2, cmap='gray')
plt.show()

# computing the epipole e_prime
#F = F.astype('float64')
L, D, R = np.linalg.svd(np.transpose(F_gt))
e_prime = R[R.shape[0]-1, :]

#Verifying if the epipole is found on the epipolar line
x = np.dot(e_prime, l_prime)
print('The dot product of epipole and epipolar line of the second image is:', x)
