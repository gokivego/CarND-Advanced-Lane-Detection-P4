# Class to calibrate the camera and return an undistorted image

import glob
import numpy as np
import matplotlib.image as mpimg
import cv2

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

                #         # draw and display the corners
                #         img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
                #         plt.imshow(img)
                #         plt.show()

        found, self.camera_matrix, self.distortion_coeff, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape,
                                                                                     None, None)

    def undistort(self, image):
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeff, None, self.camera_matrix)

    def __call__(self, image):
        return self.undistort(image)