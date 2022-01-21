import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# analyze Greyscale histogram, return list of mean, peak, area, and area of 
# selected Grey
def histGrey(image):
   # # Get Histogram [Greyscale]
      gry = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
      gry = cv.GaussianBlur(gry, (19, 19), 0)

      vals, bins, _ = plt.hist(gry.ravel(), bins=255, range=(0, 255), fc='k', ec='k')

      mask_gry = bins <= 80
      vals_gry = vals[mask_gry[:-1]]

      # Metrics of Interest
      area = sum(np.diff(bins)*vals)

      # Percent area above 200 (lighter pixels)
      area_gry = sum(vals_gry) / area
      peak = np.argmax(vals)
      mean = np.average(gry)

      return [mean, peak, area, area_gry]

def histColor(image):
      ######## HISTOGRAM METRICS [HSV]
      hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
      hsv = cv.GaussianBlur(hsv, (19,19), 0)

      bins = 180
      hist = cv.calcHist([hsv], [0], None, [bins], [0, bins])

      mean = np.average(hsv)
      peak = np.argmax(hist)
      area = np.sum(hist)

      # Starting and stopping conditions for picking off orange
      area_col = np.sum(hist[10:50]) / area
      
      return [mean, peak, area, area_col]

def shape(contour):
   # Contour Features
   # for extremely small contours, area calculations return errors
   try:
      M = cv.moments(contour)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])

      area = cv.contourArea(contour)
      per = cv.arcLength(contour, True)

      # Contour Properties
      _, _, w, h = cv.boundingRect(contour)
      # Rectangular characteristic
      aspectRatio = float(w) / h
      
      # proportion of area filled by bounding rectangle
      area_rect = w * h
      extent = float(area) / area_rect

      #ratio of contour area to convex hull
      hull = cv.convexHull(contour)
      solidity = float(area) / cv.contourArea(hull)

      #fit elipse, major minor axis, orientation angle
      (_, _), (_, _), angle = cv.fitEllipse(contour)

      return [per, area, aspectRatio, cx, cy, extent, solidity, angle]
   except :
      return[-1, -1, -1, -1, -1, -1, -1, -1]

if __name__ == '__main__':
   # Add Pickel Calls here pd.DataFrameToPickle
   print("Extraction call as main")