import os 
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import csv

dir_rd = 'C:/git/UW/581-ml/proj/data/test/'
dir_wr = 'C:/git/UW/581-ml/proj/data/curr/'

data = []

os.chdir(dir_rd)
for pathname in os.listdir(dir_rd) : 
   filename = pathname.split('/')[-1]
   os.chdir(dir_rd)
   im = cv.imread(filename)
   output = im.copy()

   # HSV Color Space
   hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
   hsv = cv.GaussianBlur(hsv, (7, 7), 0)

   # Orange Color Detection
   org = cv.inRange(hsv, (10, 50, 20), (30, 255, 255))

   # Dilate - Define a filter (kernal) to process using dilation
   krn_dil = np.ones((20, 20), np.uint8)
   dil = cv.morphologyEx(org, cv.MORPH_CLOSE, krn_dil)

   contours, hier = cv.findContours(dil, 
               cv.RETR_EXTERNAL, 
               cv.CHAIN_APPROX_SIMPLE)

   if(len(contours) == 0):
      print("Error on {}", filename)
   #   continue

   os.chdir(dir_wr)
   i = 0
   head = ['filename', 'contourBounds']
   # Select Contour
   cnt = max(contours, key=cv.contourArea)
   for cnt in contours :
      x, y, w, h = cv.boundingRect(cnt)
      data.append((filename, x, y, w, h))
      tmp = output[y:y + h, x:x + w]
      cv.drawContours(output, [cnt], 0, (0, 255, 0), 2)

      # cv.imwrite(str(filename[:-4] + "_" + str(i) + '.png'), tmp)
      i += 1

   cv.imshow("",output)
   cv.waitKey(0)

# Write Crop information to file
with open('colorDetect.txt', 'w', newline="") as f :
   write = csv.writer(f)
   # write.writerow(head)
   write.writerows(data)