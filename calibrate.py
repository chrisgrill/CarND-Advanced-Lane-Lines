import cv2
import numpy as np
import glob
import pickle
import matplotlib.pyplot as pl
import matplotlib.image as mpimg

images = glob.glob("camera_cal/calibration*.jpg")
objpoints = []
imgpoints = []
gray = []
for fname in images:
    img = cv2.imread(fname)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

img = cv2.cvtColor(cv2.imread("test_images/test1.jpg"), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
pickle.dump(mtx, open("mtx.p", "wb"))
pickle.dump(dist, open("dist.p", "wb"))
dst = cv2.undistort(img, mtx, dist, None, mtx)
pl.figure()
pl.imshow(dst)
pl.figure()
pl.imshow(img)
pl.show()






