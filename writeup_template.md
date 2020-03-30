## Writeup  - Advanced Lane Lines Finding
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/camera_cal_results1.PNG "Undistorted Camera Calibration Images"
[image2]: ./camera_cal/camera_cal_results2.PNG "Undistorted Camera Calibration Images"
[image3]: ./camera_cal/camera_cal_results3.PNG "Undistorted Camera Calibration Images"
[image4]: ./camera_cal/camera_cal_results4.PNG "Undistorted Camera Calibration Images"
[image5]: ./test_images/test2.jpg "Original Image"
[image6]: ./output_images/warped_test2.jpg "Undistorted Image"
[image7]: ./test_images/test2_res_fit.png "Color & Gradient Thresholding"
[image8]: ./output_images/around_test2.jpg "Fitting Polynomial and Searching around the fitted polynomial"
[image9]: ./test_images/test2_res_org.png "Back to the Origianl Image with lane lines"
[image10]: ./output_images/window_test2.jpg "Sliding Window Search"
[video1]: ./project_video_output.mp4 "Original Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. I identify the corners and detected the distored rectangele and I converted into undistored one. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

### Pipeline

#### 1. Here I showed the pipeline by using example image

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image5]

#### 2. I transformed the perspective from front view to bird-eye view by using perspective transformation by selecting specific region.

The code for my perspective transform includes a function called `cal_warped()`,  The `cal_warped()` function takes as inputs an image (`image`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
 src = np.float32([(600,450),
                  (700,450), 
                  (300,650), 
                  (1000,650)])
dst = np.float32([(550,0),
                  (img_size[0]-350,0),
                  (550,img_size[1]),
                  (img_size[0]-350,img_size[1])])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4.  Below it could be seen that, color transformation and gradient thresholded binary version with original image. I identified lane-line pixels by sliding window and search around polynom method.

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this: 
1. Lane lines with sliding window
2. Lane lines via searching around polynom

![alt text][image10]
![alt text][image8]

#### 5.I fit their positions with a polynomial. Then, I converted them from pixels to meters. I followed the steps in the lecture and found the curvature and lateral distance to the lane center.

I did this in lines 273 through 293 in my code in `P2.py`

#### 6. The example image result plotted back down onto the road such that the lane area is identified clearly.

Here is the original image with the lane line findings via following each step in pipeline.

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail in the image frames to process whole pipeline. I am debugging further.
