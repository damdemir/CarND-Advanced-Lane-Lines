# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 09:00:03 2020

@author: Damla Demir Yildirim
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

#Camera Calibration and perpective transformation will be done by cal_warped function
def cal_warped(image,mtx,dist):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img_size = (img.shape[1], img.shape[0])
    im_undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    src = np.float32([(600,450),
                  (700,450), 
                  (300,650), 
                  (1000,650)])
    dst = np.float32([(550,0),
                  (img_size[0]-350,0),
                  (550,img_size[1]),
                  (img_size[0]-350,img_size[1])])
    M = cv2.getPerspectiveTransform(src, dst)
    inv_M = cv2.getPerspectiveTransform(dst, src)    
    warped = cv2.warpPerspective(im_undist, M, img_size)
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,5))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(img) #original image
    ax2.imshow(im_undist) #undistorted image
    ax3.imshow(warped) # warped image
    
    return warped, M, inv_M  #this will return straight view image, perspective Mtx and inverse Mtx

def pipeline(warped, s_tresh, mag_thresh):
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    sxbinary_mag = np.zeros_like(scaled_sobel)
    sxbinary_mag[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    hls = cv2.cvtColor(warped,cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]    
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    combined_binary = np.zeros_like(sxbinary_mag)
    combined_binary[(s_binary == 1) | (sxbinary_mag == 1)] = 1
   
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(warped)
    ax2.imshow(gray, cmap='gray')
    ax3.imshow(combined_binary, cmap='gray')   
    
    return combined_binary

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin  
        win_xright_low = rightx_current - margin 
        win_xright_high = rightx_current + margin 
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img,left_lane_inds,right_lane_inds


#find region of interest, threhold and fit polynom
def fit_polynomial(combined_binary):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img, left_lane_inds, right_lane_inds = find_lane_pixels(combined_binary)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, combined_binary.shape[0]-1, combined_binary.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    return out_img, left_fit, right_fit, left_lane_inds, right_lane_inds

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    
    f, (ax1) = plt.subplots(1, figsize=(10,5))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(result) #search around poly image    
    # Plot the polynomial lines onto the image

    left_fit = np.polyfit(ploty, left_fitx, 2)
    right_fit = np.polyfit(ploty, right_fitx, 2)
    
    return result, left_fit, right_fit, ploty, left_lane_inds, right_lane_inds

def measure_curvature_pixels(left_fit, right_fit, ploty):
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
 
    return left_curverad, right_curverad

def measure_curvature_real(combined_binary, left_fit_cr, right_fit_cr, ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    h = combined_binary.shape[0]    
    y_eval = np.max(ploty)    
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)/(np.absolute(2*left_fit_cr[0]))  ## Implement the calculation of the left line here
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)/(np.absolute(2*right_fit_cr[0]))  ## Implement the calculation of the right line here
    
    if right_fit_cr is not None and left_fit_cr is not None:
        car_position = combined_binary.shape[1]/2
        l_fit_x_int = left_fit_cr[0]*h**2 + left_fit_cr[1]*h + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0]*h**2 + right_fit_cr[1]*h + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    
    return left_curverad, right_curverad, center_dist

def draw_lane_real(img, combined_binary, l_fit, r_fit, Minv):
    new_img = np.copy(img)
    if l_fit is None or r_fit is None:
        return img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = combined_binary.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result, ploty

def add_data_real(img, curverad, center_dist):
    img_w_data = np.copy(img)
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Radius of curvature = ' + '{:04.2f}'.format(curverad) + 'm'
    cv2.putText(img_w_data, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = 'Vehicle is ' + '{:04.3f}'.format(abs_center_dist) + 'm ' + direction +' of the center'
    cv2.putText(img_w_data, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return img_w_data

#object points look like (1,1,0) (6,5,0)
objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []
images = glob.glob('camera_cal/calibration*.jpg')
nx = 9
ny = 6

for fname in images:
    #print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    #img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
    #plt.imshow(img)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners) 
        ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        #Chessboard images distortion correction
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        src = np.float32([corners[0],corners[nx-1],corners[-1],corners[-nx]])
        offset = 100 # reasonable offset
        dst = np.float32([[offset, offset],[img_size[0]-offset, offset],[img_size[0]-offset, img_size[1]-offset],[offset, img_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, M, img_size)
        
        #Visualization of Camera Calibration
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,5))
        f.subplots_adjust(hspace = .2, wspace=.05)
        ax1.imshow(img) #original image
        ax2.imshow(undist) #undistorted image
        ax3.imshow(warped) # warped image
        

test_imgs = glob.glob('test_images/test*.jpg')  
#for test_im in test_imgs:
test_im = test_imgs[2]
test_im = cv2.imread(test_im)
warped, M, invM = cal_warped(test_im, mtx,dist)
s_thresh = (170,255)
mag_thresh = (30,100)
combined_binary = pipeline(warped,s_thresh,mag_thresh)

out_img, left_fit, right_fit, left_lane_inds, right_lane_inds = fit_polynomial(combined_binary)
result, left_fit, right_fit, ploty, left_lane_inds, right_lane_inds = search_around_poly(combined_binary,left_fit,right_fit)

left_curverad_pix, right_curverad_pix = measure_curvature_pixels(left_fit, right_fit, ploty)
print(left_curverad_pix, right_curverad_pix)
left_curverad_real, right_curverad_real, center_dist = measure_curvature_real(combined_binary, left_fit, right_fit, ploty)
print(left_curverad_real, 'm', right_curverad_real, 'm', center_dist, 'm')


test_im_lane, ploty = draw_lane_real(test_im, combined_binary, left_fit, right_fit, invM)
plt.imshow(test_im_lane)

test_im_data = add_data_real(test_im_lane, (left_curverad_real+right_curverad_real)/2, center_dist)
plt.imshow(test_im_data)

def process_image(test_im):
    warped, M, invM = cal_warped(test_im, mtx,dist)
    s_thresh = (170,255)
    mag_thresh = (30,100)
    combined_binary = pipeline(warped,s_thresh,mag_thresh)
    
    out_img, left_fit, right_fit, left_lane_inds, right_lane_inds = fit_polynomial(combined_binary)
    result, left_fit, right_fit, ploty, left_lane_inds, right_lane_inds = search_around_poly(combined_binary,left_fit,right_fit)
    
    left_curverad_pix, right_curverad_pix = measure_curvature_pixels(left_fit, right_fit, ploty)
    #print(left_curverad_pix, right_curverad_pix)
    left_curverad_real, right_curverad_real, center_dist = measure_curvature_real(combined_binary, left_fit, right_fit, ploty)
    #print(left_curverad_real, 'm', right_curverad_real, 'm', center_dist, 'm')
    
    
    test_im_lane, ploty = draw_lane_real(test_im, combined_binary, left_fit, right_fit, invM)
    #plt.imshow(test_im_lane)
    
    test_im_data = add_data_real(test_im_lane, (left_curverad_real+right_curverad_real)/2, center_dist)
    #plt.imshow(test_im_data)
    
    return test_im_data

from moviepy.editor import VideoFileClip
#from IPython.display import HTML

video_output1 = 'project_video_output.mp4'
video_input1 = VideoFileClip('project_video.mp4')
processed_video = video_input1.fl_image(process_image)
processed_video.write_videofile(video_output1, audio=False)
video_input1.close()
