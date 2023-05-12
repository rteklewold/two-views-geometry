import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

# Load the two images
img1 = cv2.imread('/home/ecn/lab2_avg/images_lab2/chapel00.png',cv2.IMREAD_GRAYSCALE)     # queryImage
img2 = cv2.imread('/home/ecn/lab2_avg/images_lab2/chapel01.png',cv2.IMREAD_GRAYSCALE) # trainImage
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3),plt.show()

# Load the Fundamental Matrix
F=np.loadtxt('chapel.00.01.F')
print(F)

plt.show()
