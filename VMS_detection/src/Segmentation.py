# This class allows for the extraction of the 3 largest contours by area using
#  varying segmentation methods. This included the use of adaptive thresholding
#  for edge detection, or specific extraction via color selection. Additionally
#  it allows for the user to manually classify whether the contour of interest
#  contains a VMS or not with a queried prompt

import cv2 as cv
import numpy as np

def blur(image, kernel_size = 7):
   # Slight blur to remove noise and get cleaner edges

    im = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return im

# get contour information based on adaptive thresholding wiht a mean shift
def greyGetContour(image) :
    # Slight blur to remove noise and get cleaner edges
    im = blur(image)

    # gry - edge detection
    gry = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    gry[gry > 80] = 255

    # drop pixels down to 0 min
    gry = gry - np.min(gry)

    # remove max values to mean for thresholding
    gry[gry == np.max(gry)] = np.average(gry)
    gry = np.array(gry, np.uint8)

    # subtract 1 sigma from the mean
    thresh = cv.adaptiveThreshold(gry, 255, 
                                 cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY_INV,
                                 101,
                                 np.average(gry) * 0.08)

    # # Opening - Define a filter (kernal) to process using opening (3x3)
    # adaptive kernel based on image size?
    krn = np.ones((20, 20), np.uint8)

    # getStructuringElement - Close image, then open
    close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, krn)
    open_ = cv.morphologyEx(close, cv.MORPH_OPEN, krn)
    mask = cv.bitwise_and(open_, close)
    # mask = close

    return __getContours(mask)

# Get contour selection based on color thresholding - identify orange/yellow
def colorGetContour(image) :
    # HSV Color Space
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hsv = blur(hsv)

    # Orange Color Detection
    org = cv.inRange(hsv, (10, 50, 20), (30, 255, 255))

    # Dilate - Define a filter (kernal) to process using dilation
    krn_dil = np.ones((20, 20), np.uint8)
    dil = cv.morphologyEx(org, cv.MORPH_CLOSE, krn_dil)

    return __getContours(dil)

# private method of retrieving contours of an image, returns 3 largest contours
def __getContours(img):
    contours, _ = cv.findContours(img, 
               cv.RETR_EXTERNAL, 
               cv.CHAIN_APPROX_NONE)

    if(len(contours) == 0):
        return []
    else : 
      # Return 3 largest contours
        cnt = sorted(contours, key = cv.contourArea)
        return cnt[-3:]

# public method for retrieving contours, either all, only color, or only thresh
def getContours(IM):
   cnt_list = colorGetContour(IM) + greyGetContour(IM)
   cnt_list = sorted(cnt_list, key=cv.contourArea)
   return cnt_list[-3:]

# when a contour is dislayed to the user, manually input if contour contains
# VMS or not for storage
def labelContour():
    val = bool(input("Input any char if displayed contour contains VMS\nOtherwise, press enter to continue:"))
    if val:
        return [1]
    else:
        return [0]