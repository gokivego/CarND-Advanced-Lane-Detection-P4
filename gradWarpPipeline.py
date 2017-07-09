import cv2
import numpy as np
import matplotlib.pyplot as plt

# This file calculates the different gradients and sobel thresholds and finds the lanes. Also returns a warped image

def plotImages(img1, title1, img2, title2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=50)
    ax2.imshow(img2)
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

class processingPipeline(object):
    def __init__(self, kernal=3, orient='x'):

        self.sobel_kernel = 3

        self.absolute_threshold = (20, 100)
        self.magnitude_threshold = (20, 100)
        self.direction_threshold = (0.7, 1.3)
        self.color_threshold = (170, 255)

        self.s_channel = 2
        self.r_channel = 0

    def __abs_sobel_thresh(self, img, sobel_kernal=3, orient='x', threshold=(0, 255)):
        # use the red channel of the image as in the videos that turned out to be the best to identify the lanes
        # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
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

    def __color_thresh(self, img, threshold=(0, 255)):
        # color thresholding on the color intensity threshold

        binary_output = np.zeros_like(img)
        binary_output[(img > threshold[0]) & (img <= threshold[1])] = 1

        # plotImages(img,'Original Road Image', binary_output, 'Color Thresholded Image')

        return binary_output

    def plot_abs_sobel_thresh(self, img):
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        hls_schannel = hls_img[:, :, self.s_channel]
        plotImages(img, 'Original Road Image', self.__abs_sobel_thresh(hls_schannel, orient='x', threshold=self.absolute_threshold), 'Thresholded x-derivative')

    def plot_mag_sobel_thresh(self, img):
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        hls_schannel = hls_img[:, :, self.s_channel]
        plotImages(img, 'Original Road Image', self.__mag_sobel_thresh(hls_schannel, threshold=self.magnitude_threshold), 'Thresholded Magnitude')

    def plot_dir_sobel_thresh(self, img):
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        hls_schannel = hls_img[:, :, self.s_channel]
        plotImages(img, 'Original Road Image', self.__dir_sobel_thresh(hls_schannel, threshold=self.direction_threshold), 'Thresholded Grad.Dir')


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


class perspective_transform(object):
    # Unwarps the image to get a birds eye view image
    def __init__(self):
        src = np.float32([
            [580, 460],
            [700, 460],
            [1040, 680],
            [260, 680],
        ])

        dst = np.float32([
            [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720],
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def unwarp(self, image):
        return cv2.warpPerspective(image, self.M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def __call__(self, image, unwarp=False):
        if unwarp:
            return self.unwarp(image)
        else:
            return self.warp(image)


class gradWarpPipeline(object):

    def __call__(self, image, stacked=False):
        bin_output = processingPipeline()(image, stacked)
        return perspective_transform()(bin_output)