# Advanced Lane Finding Project

---

### The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images_for_writeup/1.chessboard_undistorted.png "Undistorted Chess board Image"
[image2]: ./images_for_writeup/2.undistorted_road.png "Undistorted Road Image"

[image3]: ./images_for_writeup/3.gradient_abs_value.png "Gradient Absolute Value"
[image4]: ./images_for_writeup/3.gradient_magnitude.png "Gradient Magnitude"
[image5]: ./images_for_writeup/3.gradient_direction.png "Gradient Direction"
[image6]: ./images_for_writeup/3.final_binary.png "Final Binary"

[image7]: ./images_for_writeup/4.warped_straight_lines.jpg "Warp Example"
[image8]: ./images_for_writeup/5.hist_plot.png "Histogram Peaks"
[image9]: ./images_for_writeup/6.windows.png "Sliding Window"
[image10]: ./images_for_writeup/7.rcurve.png "Equation"
[image11]: ./images_for_writeup/8.final_image_with_lanes.png "Final"

[video1]: ./project_video_annotated_vego.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

### 1.Camera Calibration

Cameras generate distorted images. The type of distortion that the camera generates is dependent upon the lens present in the camera. So, images are especially stretched or skewed in the corners of the camera images.

The primary reason for image distortion is because a camera at a 3D object in the real world but what we are lookign at is a 2D image. In the process of this transformation distortion is induced in the image and we need to correct for this distortion if we are to find the lane lines accurately.

Cameras typically use curved lenses and when light passes through the camera it bends a little too much or too little at the edges and this type of distortion is called `radial distortion` and is the most common form of distortion. To correct for this:


I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Here is my code and the result

    class calibrateCamera(object):

    def __init__(self, calibration_images_folder, corners=(8, 6)):

    # Read in and make a list of calibration images
    self.images = glob.glob(calibration_images_folder + 'calibration*.jpg')
    self.corners = corners

    # The camera matrix and distortion coefficients returned by cv2.calibrateCamera function

    self.camera_matrix = None
    self.distortion_coeff = None
    self.__calibrate()

    def __calibrate(self):

    # Arrays to store object points and image points from all the images

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image space

    # Prepare object points, like (0,0,0), (1,0,0) etc
    objp = np.zeros([self.corners[0] * self.corners[1], 3], np.float32)
    objp[:, :2] = np.mgrid[0:self.corners[0], 0:self.corners[1]].T.reshape(-1,
    2)  # creating the x, y coordinates for the corners

    for fname in self.images:
    # read in each image
    img = mpimg.imread(fname)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    found, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # if corners are detected, then append the object points and image points

    if found == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    found, self.camera_matrix, self.distortion_coeff, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape,
    None, None)

    def undistort(self, image):
    return cv2.undistort(image, self.camera_matrix, self.distortion_coeff, None, self.camera_matrix)

    def __call__(self, image):

![alt text][image1]

### Pipeline (single images)

#### 1.Example of a Distortion Correct Image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

In this image we can observe a distortion correction. Look carefully at the white car in both the images, distorted and undistorted. In the undistorted image the car is closer to the one of the edges. 

#### 2. Thresholding Methods to identify Lanes 

I used a combination of color and gradient thresholds to generate a binary image. 
    
#### Gradient Absolute Value

    def __abs_sobel_thresh(self, img, sobel_kernal=3, orient='x', threshold=(0, 255)):
        #use the red channel of the image as in the videos that turned out to be the best to identify the lanes
        #Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
        if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernal))
        if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernal))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1

        # plotImages(img, 'Original Road Image', binary_output, 'Thresholded '+ orient + '-derivative')
        # Return the result
        return binary_output

![alt text][image3]

#### Gradient Magnitude

        # Returns the magnitude of the gradient
        # for a given sobel kernel size and threshold values

    def __mag_sobel_thresh(self, img, sobel_kernel=3, threshold=(0, 255)):
        # Applying on red channel
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= threshold[0]) & (gradmag <= threshold[1])] = 1

        # plotImages(img,'Original Road Image', binary_output, 'Thresholded Magnitude')

        # Return the binary image
        return binary_output

![alt text][image4]

#### Gradient Direction 

    def __dir_sobel_thresh(self, img, sobel_kernel=3, threshold=(0, np.pi / 2)):
        # Applying on red channel
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the absolute value of the x and y gradients and calculating the direction of the gradient
        direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(direction)
        binary_output[(direction >= threshold[0]) & (direction <= threshold[1])] = 1

        # plotImages(img,'Original Road Image', binary_output, 'Thresholded Grad.Dir')

        # Return the binary image
        return binary_output

![alt text][image5]

#### Color Threshold

    def __color_thresh(self, img, threshold=(0, 255)):
        # color thresholding on the color intensity threshold

        binary_output = np.zeros_like(img)
        binary_output[(img > threshold[0]) & (img <= threshold[1])] = 1

        # plotImages(img,'Original Road Image', binary_output, 'Color Thresholded Image')

#### Applying the above thresholding methods and creating a binary image

    def get_processed_image(self, img, stacked=False):

        # Processes the images and returns the image in a form thats best identifies the lanes
        #  First converting the image to HLS color channel
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        hls_schannel = hls_img[:, :, self.s_channel]

        # Finding the different thresholding sobel images and combining them

        sobelx = self.__abs_sobel_thresh(hls_schannel, orient='x', threshold=self.absolute_threshold)
        sobely = self.__abs_sobel_thresh(hls_schannel, orient='y', threshold=self.absolute_threshold)
        magnitude = self.__mag_sobel_thresh(hls_schannel, threshold=self.magnitude_threshold)
        direction = self.__dir_sobel_thresh(hls_schannel, threshold=self.direction_threshold)
        binary_output_grad = np.zeros_like(hls_schannel)
        binary_output_grad[((sobelx == 1) & (sobely == 1)) | ((magnitude == 1) & (direction == 1))] = 1

        # Applying color threshold on the binary_output
        binary_output_color = self.__color_thresh(hls_schannel, threshold=self.color_threshold)

        if stacked:
        stacked = np.dstack((np.zeros_like(hls_schannel), binary_output_grad, binary_output_color))
        #             plotImages(hls_schannel, 'Original Road Image',stacked, 'Final Image')
        return stacked
        else:
        binary_output = np.zeros_like(binary_output_grad)
        binary_output[(binary_output_grad == 1) | (binary_output_color == 1)] = 1
        # plotImages(hls_schannel,'hls image',binary_output,'Final Image')
        return binary_output

        def __call__(self, img, stacked=False):
        return self.get_processed_image(img, stacked)

![alt text][image6]

#### 3. Perspective Transform

The code for my perspective transform includes a function called `perspective_transform()`, The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

The following are the source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 260, 0        | 
| 700, 460      | 1040, 0       |
| 1140, 680     | 1040, 720     |
| 260, 680      | 260, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

#### 4. Lane Finding Method: Histogram Peaks

To find which pixels are part of the line and which pixels are not I took a histogram of the bottom half of the image and the position of the lanes is marked by a spike in the histogram like this:

![alt text][image8]

#### Sliding Window Approach

By adding up the pixel values along each column in the image we can find the peaks in the histogram and these peaks in the histogram will be good indicators of the x-position of the base of the lane lines. These points can be used as starting point for where to search for the lines. From here on, I used a sliding window placed around line centers to form the new windows up till the top of the frame. After that, I scanned each window to collect the non-zero pixels within window bounds. Then a second order polynomial can be fit to the collected points.  

![alt text][image9]

#### 5. Determine the lane curvature and the vehicle position with respect to the center.

We use the pixels that have been accumulated to find which pixels belong to left lane and whixh pixels belong to the right lane and then we fit a polynomial using the equation:

![alt text][image10]

    def measure_curvature(self):
        points = self.points()
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        x = points[:, 0]
        y = points[:, 1]

        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        curve_radius = ((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) \
        / np.absolute(2 * fit_cr[0])
        return int(curve_radius)

    def vehicle_position(self):
        points = self.points()
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        x = points[np.max(points[:, 1])][0]
        return np.absolute((self.width // 2 - x) * xm_per_pix)

#### 6. Final Processed Image

![alt text][image11]

---

### Pipeline (video)

The pipeline and the sliding window algorithm is applied to a sequence of frames in the video. A line is drawn by taking the average of the polynomial coefficients detected over the last 5 frames.

Here's a [link to my video result](./project_video_annotated_vego.mp4)

![alt text][video1]

---

#### Discussion

I took the procedure discussed in the course videos and implemented the pipeline to detect the lanes. However I observed that in low light conditions the pipeline doesn't perform so well. The pipeline didn't perform so well on the harder video challenge. The video also involved tuning various hyper parameters to extract the lines in the video.

#### Reflections

I had a lot of fun working on the project. I used the techniques of computer vision in the implementation of the project and I believe incorporating techniques of machine learning in the pipeline could bring great improvement in the project.

