import cv2
import numpy as np
import glob
import pickle
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import platform


mtx = pickle.load(open("mtx.p", "rb"))
dist = pickle.load(open("dist.p", "rb"))

offset = 100
img = cv2.imread("test_images/test6.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]
img_size = (gray.shape[1], gray.shape[0])
src = np.float32([[220,719],[1220,719],[750,480],[550,480]])
dst = np.float32([[240,719],[1040,719],[1040,300],[240,300]])
thresh = (90, 255)
binary = np.zeros_like(S)
binary[(S > thresh[0]) & (S <= thresh[1])] = 1
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(binary, M, img_size)
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)


pl.figure()
pl.plot(histogram)
pl.figure()
pl.imshow(warped, cmap='gray')
pl.figure()
pl.imshow(S, cmap='gray')
pl.show()


