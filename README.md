
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

[image1]: ./output_images/original_image.jpg "Original Image"
[image2]: ./output_images/undistorted_image.jpg "Undistorted Image"
[image3]: ./output_images/Original_lanes.jpg "Original Lanes Image"
[image4]: ./output_images/Undistored_lanes.jpg "Undistorted Lanes Image"
[image5]: ./output_images/lanes_bird_view.jpg "Lanes bird view"
[image6]: ./output_images/tracked_Lanes.jpg "Tracked Lanes"
[image7]: ./output_images/lane_fitting.jpg "Lane fitting"
[image8]: ./output_images/color_fit_lanes.jpg "Fit Visual"
[image9]: ./output_images/Finaloutput.jpg "Output"
[video10]: ./project_output_final.mp4 "Video"


---
###README

###Camera Calibration

The code for this step is in `calibrate()` of LaneTracker.py

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

###Pipeline (single images)

Now I applied distortion correction function to the images from the video clips and results are as shown below:
###Original image
![alt text][image3]

###Undistorted image
![alt text][image4]

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps can be found in `Process_data_mag_abs_color()` in `LaneTracker.py`).  Here's an example of my output for this step. 

![alt text][image6]

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 136 through 147 in the file `LaneTracker.py`.  The `perspective_transform()` function takes as inputs an image (`img`), as well as size_top, size_bottom of the trapeziod required for lane marking.  source (`src`) and destination (`dst`) points are calculated using the formaule mentioned below.

```
src = np.float32(
    [[(width/2) - size_top, height*0.65],
    [(width/2) + size_top, height*0.65],
    [(width/2) + size_bottom, height-50],
    [(width/2) - size_bottom, height-50]])
dst = np.float32(
    [[(width/2) - output_size, (height/2) - output_size],
    [(width/2) + output_size, (height/2) - output_size],
    [(width/2) + output_size, (height/2) + output_size],
    [(width/2) - output_size, (height/2) + output_size]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 468      | 280, 0        | 
| 710, 468      | 1000, 720     |
| 1010, 670     | 1000, 720     |
| 270, 670      | 280, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart by drawing the bird's view as shown below

![alt text][image5]

Then I fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image7]

The radius of curvature is calculated using the `get_curvature()`. and the offset is calculate in process_image() in `LaneTracker.py`

I then rendered the path using the dtected lane lines like the image shown below using the `render_lane_detected()`:

![alt text][image6]

and the Final Image Looks like this.
![alt text][image9]
---

###Pipeline (video)

Here's a [link to my video result](./project_output_final.mp4)

---
###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline will fail if the lane lines become less visible due to high sunshine on road or dark shadows. Also need to work on video stabilization before processing. Because the Bumpiness of the road will also make the prediction of lane lines difficult.  


