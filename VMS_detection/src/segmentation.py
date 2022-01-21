import cv2 as cv
import numpy as np

def getContourThresh(image) :
   # Slight blur to remove noise and get cleaner edges
   im = cv.GaussianBlur(image, (7, 7), 0)

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


def getContourColor(image) :
   # HSV Color Space
   hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
   hsv = cv.GaussianBlur(hsv, (7, 7), 0)

   # Orange Color Detection
   org = cv.inRange(hsv, (10, 50, 20), (30, 255, 255))

   # Dilate - Define a filter (kernal) to process using dilation
   krn_dil = np.ones((20, 20), np.uint8)
   dil = cv.morphologyEx(org, cv.MORPH_CLOSE, krn_dil)

   return __getContours(dil)

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


def getContours(IM, method = 'threshold'):
   if method == 'color':
      return getContourColor(IM)
   else:
      return getContourThresh(IM)