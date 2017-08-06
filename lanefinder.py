import cv2
import numpy as np


class LaneFinder:

    def __init__(self):
        self.window_image = None
        self.leftx_base = 0
        self.rightx_base = 0
        self.left_fitx = None
        self.right_fitx = None
        self.ploty = None

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    def set_base(self, histogram):
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    def binary(self, img, thresh=(90, 255)):
        binary = np.zeros_like(img)
        binary[(img > thresh[0]) & (img <= thresh[1])] = 1
        return binary

    def warp(self, img):
        s_channel = self.s_channel(img)
        sobel = self.sobel(img)
        img_size = (img.shape[1], img.shape[0])
        src = np.float32([[220, 719], [1220, 719], [750, 480], [550, 480]])
        dst = np.float32([[240, 719], [1040, 719], [1040, 300], [240, 300]])
        binary_s = self.binary(s_channel)
        binary_sobel = self.binary(sobel)
        binary = cv2.bitwise_or(binary_s, binary_sobel)
        transform_matrix = cv2.getPerspectiveTransform(src, dst)
        return s_channel, cv2.warpPerspective(binary, transform_matrix, img_size), transform_matrix

    def s_channel(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        return s_channel

    def sobel(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        return scaled_sobel