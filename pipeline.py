import cv2
import numpy as np
import glob
import pickle
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import platform
from line import Line
from lanefinder import LaneFinder

right_line = Line()
left_line = Line()
lf = LaneFinder()
# Load camera calibration data
mtx = pickle.load(open("mtx.p", "rb"))
dist = pickle.load(open("dist.p", "rb"))
cap = cv2.VideoCapture("project_video.mp4")
frame_num = 0
l_prev_fit = []
r_prev_fit = []
pre_left_fitx = []
pre_right_fitx = []
pre_ploty = []
pre_right_slope = 0
pre_left_slope = 0
pre_lane_width = 0
while(True):
    print("Frame: "+str(frame_num))
    ret, img = cap.read()
    #img = cv2.imread("output_images/frame001040.jpg")
    # Undistort using camera calibration data
    img = cv2.undistort(img, mtx, dist, None, mtx)
    # Perform perspective transform and return binary image ideally containing lane lines
    s_channel, warped, transform_matrix = lf.warp(img)
    # Get histogram across center of image
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    lf.out_img = np.dstack((warped, warped, warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 20
    # Set height of windows
    lf.window_height = np.int(warped.shape[0]/nwindows)
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    lf.margin = 75
    # Set minimum number of pixels found to recenter window
    lf.minpix = 20
    # Get indices of lane pixels
    left_lane_inds, right_lane_inds = lf.window_search(warped, nwindows,leftx_current, rightx_current)

    # Extract left and right line pixel positions
    leftx = lf.nonzerox[left_lane_inds]
    lefty = lf.nonzeroy[left_lane_inds]
    rightx = lf.nonzerox[right_lane_inds]
    righty = lf.nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    smoothing = 1.0
    left_fit = np.polyfit(lefty, leftx, 2)
    if len(l_prev_fit) > 0:
        left_fit = smoothing * left_fit + (1 - smoothing) * l_prev_fit
        l_prev_fit = left_fit

    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        if len(r_prev_fit) > 0:
            right_fit = smoothing * right_fit + (1 - smoothing) * r_prev_fit
            r_prev_fit = right_fit

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Sanity Checks

    left_slope = (ploty[719]-ploty[0])/(left_fitx[719] - left_fitx[0])
    right_slope = (ploty[719] - ploty[0]) / (right_fitx[719] - right_fitx[0])
    lane_width = right_fitx[719] - left_fitx[719]
    print(np.abs(left_slope- right_slope))
    if (np.abs(left_slope - right_slope) > 3 and np.abs(left_slope - right_slope) < 100) or \
            (np.abs(lane_width-pre_lane_width) > 100 and pre_lane_width > 0)\
            or (np.abs(pre_right_slope - right_slope) > 1 and pre_right_slope > 0):
        left_fitx = pre_left_fitx
        right_fitx = pre_right_fitx
        ploty = pre_ploty
        pre_right_slope = right_slope
    else:
        pre_left_fitx = left_fitx
        pre_right_fitx = right_fitx
        pre_ploty = ploty
        pre_left_slope = left_slope
        pre_right_slope = right_slope
        pre_lane_width = lane_width


    left_pts = np.column_stack((left_fitx, ploty))
    right_pts = np.column_stack((right_fitx, ploty))
    lane_center = ((np.argmin(right_pts[0]) - np.argmin(left_pts[0])) / 2) + leftx_base
    lf.out_img[lf.nonzeroy[left_lane_inds], lf.nonzerox[left_lane_inds]] = [255, 0, 0]
    lf.out_img[lf.nonzeroy[right_lane_inds], lf.nonzerox[right_lane_inds]] = [0, 0, 255]
    cv2.polylines(lf.out_img, [np.int32(left_pts)], 0,(0, 255, 255))
    cv2.polylines(lf.out_img, [np.int32(right_pts)], 0, (0, 255, 255))

    offset = lane_center - lf.out_img.shape[0]
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    vehicle_pos = offset * xm_per_pix

    # Fit new polynomials to x,y in world space
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    if len(rightx) > 0:
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))


    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))


    # Warp the blank back to original image space using inverse perspective matrix
    newwarp = cv2.warpPerspective(color_warp,np.linalg.inv(transform_matrix), (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(lf.out_img, ''.join([str(left_curverad), 'm, ', str(right_curverad), 'm']), (10, 700), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(lf.out_img, str(vehicle_pos) + 'm', (10, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    frames = np.concatenate((result,lf.out_img), axis = 1)
    h, w = int(frames.shape[0]/2), int(frames.shape[1]/2)
    frames = cv2.resize(frames, (w, h))
    s_channel = cv2.resize(s_channel, (w, h))
    cv2.imshow("Schannel", s_channel)
    cv2.imshow("Frames", frames)
    #cv2.waitKey()
    #cv2.imwrite("output_images/frame"+ str(frame_num).zfill(6) + ".jpg",img)
    frame_num = frame_num + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pl.plot(np.double(histogram))
        pl.show()
        break



