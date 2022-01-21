import os 
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

dir_rd = 'C:/git/UW/581-ml/proj/data/test/'
dir_wr = 'C:/git/UW/581-ml/proj/data/curr/'

os.chdir(dir_rd)
# Copy original image for output cropping
im = cv.imread("0003.jpg")
output = im

os.chdir(dir_wr)

 # Slight blur to remove noise and get cleaner edges
# im = cv.GaussianBlur(im, (2, 2), 0)

# hsv - color detection
hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)

# gry - edge detection
gry = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# Log Image Transformation - Enhance DArker pixels

imPow = np.power(1.1, gry)

# cap pixels at 255
imPow[imPow > 255] = 255
imPow[imPow < 255] = 0

# # remove max values to mean for thresholding
# imPow[imPow == np.max(imPow)] = np.average(imPow)

imPow = np.array(imPow, np.uint8)
cv.imshow("Power", imPow)

krn = np.ones((2, 1), np.uint8)

# Erosion on Power?
open = cv.morphologyEx(imPow, cv.MORPH_OPEN, krn)
cv.imshow("Eroded", open)

cv.waitKey(0)
cv.destroyAllWindows()

# pytesseract.image_to_string(cropped)