import cv2
import numpy as np


class LaneFinder:

    def __init__(self):
        self.out_image = None
        self.leftx_base = 0
        self.rightx_base = 0
        self.left_fitx = None
        self.right_fitx = None
        self.ploty = None
        self.window_height = 0
        self.margin = 0
        self.minpix = 0
        self.nonzerox = []
        self.nonzeroy = []

    def set_base(self, histogram):
        """Find the peak of the left and right halves of the histogram. These will be the starting point for the left
        and right lines
        :param histogram: Histogram from binary warped image
        """
        midpoint = np.int(histogram.shape[0] / 2)
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    def binary(self, img, thresh=(90, 255)):
        """Return binary thresholded image.
        :param img: Image to threshold
        :param thresh: Threshold values"""
        binary = np.zeros_like(img)
        binary[(img > thresh[0]) & (img <= thresh[1])] = 1
        return binary

    def warp(self, img):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s_channel = self.s_channel(img)
        sobelx = self.sobelx(img)
        sobley = self.sobely(img)
        img_size = (img.shape[1], img.shape[0])
        src = np.float32([[220, 719], [1220, 719], [750, 480], [550, 480]])
        dst = np.float32([[240, 719], [1040, 719], [1040, 300], [240, 300]])
        binary_s = self.binary(s_channel)
        binary_sobelx = self.binary(sobelx)
        binary_sobely = self.binary(sobley)
        binary_sobel = cv2.bitwise_or(binary_sobelx, binary_sobely)
        binary = cv2.bitwise_or(binary_s, binary_sobelx)
        transform_matrix = cv2.getPerspectiveTransform(src, dst)
        return s_channel, cv2.warpPerspective(binary, transform_matrix, img_size), transform_matrix

    def s_channel(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        return s_channel

    def sobelx(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        return scaled_sobel

    def sobely(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        abs_sobely = np.absolute(sobely)
        scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))
        return scaled_sobel

    def window_search(self, warped, nwindows, leftx_current, rightx_current):
        nonzero = warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        left_lane_inds = []
        right_lane_inds = []
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window + 1) * self.window_height
            win_y_high = warped.shape[0] - window * self.window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low) & (
            self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) & (
            self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        return left_lane_inds, right_lane_inds