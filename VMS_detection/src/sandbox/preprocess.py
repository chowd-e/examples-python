import os 
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

dir_rd = 'C:/git/UW/581-ml/proj/data/test/'
dir_wr = 'C:/git/UW/581-ml/proj/data/pre/'

for i in range(1,269):
   imgNum = i
   filetype = ".jpg"

   if(imgNum < 10):
      prepend = "000"
   elif(imgNum < 100):
      prepend = "00"
   else:
      prepend = "0"

   filename = prepend + str(imgNum) + filetype

   os.chdir(dir_rd)
   im = cv.imread(filename)
   og = im

   # Slight blur to remove noise and get cleaner edges
   im = cv.GaussianBlur(im, (7, 7), 0)
   # cv.imshow("Original Image", im)

   gry = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
   # cv.imshow("Greyscale Image", gry)

   # Only identify dark areas >200
   (T, thresh) = cv.threshold(gry, 50, 255, cv.THRESH_BINARY)
   # cv.imshow("Threshold Image", thresh)

   # # Opening - Define a filter (kernal) to process using opening (3x3)
   krn = np.ones((20, 20), np.uint8)

   # getStructuringElement
   open = cv.morphologyEx(thresh, cv.MORPH_OPEN, krn)
   # cv.imshow("Eroded Image", open)

   close = cv.morphologyEx(open, cv.MORPH_CLOSE, krn)
   # cv.imshow("Dilated Image", close)

   # Find only the extreme contours
   # Store ALL points of bounding box
   close = ~close

   contours, _ = cv.findContours(close, 
                  cv.RETR_EXTERNAL, 
                  cv.CHAIN_APPROX_NONE)

   cnt = max(contours, key=cv.contourArea)


   cv.drawContours(im, [cnt], 0, (0, 255, 0), 3)
   # cv.imshow("contoured", im)

   # Mask image

   mask = cv.bitwise_and(im, im, mask = close)
   # cv.imshow("Mapped Image", mask)

   rect = cv.boundingRect(cnt)
   cropped = og[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
   # cv.imshow("Cropped", cropped)

   os.chdir(dir_wr)
   cv.imwrite(filename, cropped)
   # cv.waitKey(0)
   # cv.destroyAllWindows()


   # https://stackoverflow.com/questions/64868166/
   # binarize-bad-background-image-using-opencv-python

   # Adaptive Thresholding
   # Deals with differing lighting conditions
   # flt = cv.adaptiveThreshold(gry, 100, 
   #                             cv.ADAPTIVE_THRESH_GAUSSIAN_C,
   #                             cv.THRESH_BINARY, 13, 16)


   # # Morphological Transform
   # # Opening - Define a filter (kernal) to process using opening 
   # krn = np.ones((3, 3), np.uint8)

   # # (erosion + dilation) to remove Noise
   # open = cv.morphologyEx(flt, cv.MORPH_OPEN, krn)

   # # and closing (reverse Opening) - closes small holes in foreground obj 
   # # or small black points
   # close = cv.morphologyEx(open, cv.MORPH_CLOSE, krn)

   # Bitwise Operation
   # mask = cv.bitwise_or(im, im, mask = thresh)
   # cv.imshow("Mapped Image", mask)

