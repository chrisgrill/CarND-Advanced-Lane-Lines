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
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        img_size = (hls.shape[1], hls.shape[0])
        src = np.float32([[220, 719], [1220, 719], [750, 480], [550, 480]])
        dst = np.float32([[240, 719], [1040, 719], [1040, 300], [240, 300]])
        binary = self.binary(s_channel)
        transform_matrix = cv2.getPerspectiveTransform(src, dst)
        return s_channel, cv2.warpPerspective(binary, transform_matrix, img_size), transform_matrix

    def window_search(self, img, nwindows=9):
        # Set height of windows
        window_height = np.int(img.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(self.window_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.window_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        self.left_fitx = left_fit[0] * self.ploty ** 2 + left_fit[1] * self.ploty + left_fit[2]
        self.right_fitx = right_fit[0] * self.ploty ** 2 + right_fit[1] * self.ploty + right_fit[2]

        self.window_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        self.window_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]