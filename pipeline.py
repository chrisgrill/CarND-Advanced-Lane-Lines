import cv2
import numpy as np
import glob
import pickle
import matplotlib.pyplot as pl
import matplotlib.image as mpimg

mtx = pickle.load(open("mtx.p", "rb"))
dist = pickle.load(open("dist.p", "rb"))

offset = 100
#img = cv2.imread("test_images/test1.jpg")
img = mpimg.imread("test_images/test2.jpg")

#img = cv2.undistort(img, mtx, dist, None, mtx)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_size = (gray.shape[1], gray.shape[0])
#src = np.float32([[200,700],[700,700],[600,400],[700,400]])
src = np.float32([[220,719],[1220,719],[750,480],[550,480]])
# dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
#                              [img_size[0]-offset, img_size[1]-offset],
#                             [offset, img_size[1]-offset]])
dst = np.float32([[240,719],[1040,719],[1040,300],[240,300]])
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, img_size)
pl.figure()
pl.imshow(warped)
pl.figure()
pl.imshow(img)
pl.show()


