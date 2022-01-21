import os 
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import csv

dir_rd = 'C:/git/UW/581-ml/proj/data/test/'
dir_wr = 'C:/git/UW/581-ml/proj/data/curr/'

data = []
os.chdir(dir_rd)

for filename in os.listdir():
   os.chdir(dir_rd)
   # Copy original image for output cropping
   im = cv.imread(filename)
   output = im

   os.chdir(dir_wr)

   # Slight blur to remove noise and get cleaner edges
   im = cv.GaussianBlur(im, (7, 7), 0)

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

   contours, hier = cv.findContours(mask, 
                  cv.RETR_EXTERNAL, 
                  cv.CHAIN_APPROX_NONE)

   # use FLOODFILL to fill in like pixels / group like pixels
   # contours, hier = cv.findContours(mask,
   #                   cv.RETR_FLOODFILL,
   #                   cv.CHAIN_APPROX_SIMPLE)

   # test for empty contours and continue if needed - print to console error
   if(len(contours) == 0):
      print("Error on {}", filename)
      continue

   # Extreme Points
   i = 0
   for cnt in contours:
      # leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
      # rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
      # topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
      # bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
      data.append((filename, i))
      x, y, w, h = cv.boundingRect(cnt)
      tmp = output[y:y + h, x:x + w]
      # cv.imwrite(str(filename[:-4] + "_" + str(i) + ".png"), tmp)
      cv.drawContours(output, [cnt], 0, (0, 255, 0), 2)
      
      i += 1

   cv.imshow("", output)
   cv.waitKey(0)
   # cv.imwrite(filename, output)

   # cv.imshow("Original Image", im)
   # cv.imshow("Greyscale Image", gry)
   # cv.imwrite(str("gray_" + filename), gry)
   # cv.imshow("Threshold Image", thresh)
   # cv.imshow("Merged Image", merge)
   # cv.imshow("Dilated Image", close)
   # cv.imshow("Eroded Image", open)
   # cv.imshow("Mapped Image", mask)
   # cv.imwrite(str("merge_" + filename), mask)
   # cv.imshow("contoured", im)
   # cv.imwrite(str("contoured_" + filename), im)
   # cv.imshow("Cropped", output)
   
   # cv.waitKey(0)
   # cv.destroyAllWindows()

# Write Contour information to file
# with open('shapeDetect.txt', 'w') as f :
#    write = csv.writer(f)
#    write.writerows(data)