import cv2
import numpy as np
import pickle

mtx = pickle.load(open("mtx.p", "rb"))
dist = pickle.load(open("dist.p", "rb"))

offset = 100
nx = 9
ny = 6
img = cv2.imread("camera_cal/calibration19.jpg")
img = cv2.undistort(img, mtx, dist, None, mtx)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_size = (gray.shape[1], gray.shape[0])
ret, corners = cv2.findChessboardCorners(img, (nx,ny), None)
if ret:
        cv2.drawChessboardCorners(gray, (nx,ny), corners, ret)
        src = np.float32([corners[0],corners[nx-1],corners[-1],corners[-nx]])
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size)

cv2.imshow("Warped", img)
cv2.waitKey()

