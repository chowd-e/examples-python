# This class is designed to extract a set of features from a given image, 
# features include greyscale, color, and shape features based on contours or 
# subsets of images
import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from VmsImage import SegmentationTypeEnum, VmsImage

"""
Command Structure
"""
def getAllFeatureNames():
   return [*feature_set]

def processCommand(cmd, args = []):
   if cmd in feature_set:
      return feature_set[cmd](args)
   else:
      print("Feature not found: \"%s\"\n\nFeature List:" % cmd)
      print(*feature_set, sep ="\n")
      return -1

"""
Private
"""
def _getContourFeatures(image, contour, featureNames = []):
   features = dict()
   # ALL FEATURES
   if featureNames == []:
      for feat in feature_set.keys():
         features[feat] = processCommand(feat, [image, contour])
   # Select Features
   else:
      for feat in featureNames:
         features[feat] = processCommand(feat, [image, contour])
         
   return features   

def _getGreyscaleHistogram(image, contour):
   #  subset image based on contour bounds to retain color
   image = image.getImageSubset(contour)
   gry = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
   gry = cv.GaussianBlur(gry, (19, 19), 0)

   vals, bins, _ = plt.hist(gry.ravel(), bins=255, range=(0, 255),
                            histtype='step', fc='k', ec='k')
   plt.title("Greyscale Histogram")
   plt.show()
   return vals, bins
   
def _getHsvHistogram(image, contour):
   image = image.getImageSubset(contour)
   hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
   hsv = cv.GaussianBlur(hsv, (19,19), 0)

   bins = 180
   hist = cv.calcHist([hsv], [0], None, [bins], [0, bins])
   plt.plot(hist, color='b')
   plt.title("HSV Color Histogram")
   plt.show()
   return hist 

"""
Public
"""
def getGreyMean(args):
   if not isinstance(args[0], VmsImage) or \
      not isinstance(args[1], type(np.zeros(1))): 
      return -1      
   
   image = args[0]
   contour = args[1]
   vals, _ = _getGreyscaleHistogram(image, contour)
   return np.mean(vals)

def getGreyPeak(args):
   if not isinstance(args[0], VmsImage) or \
      not isinstance(args[1], type(np.zeros(1))): 
      return -1

   image = args[0]
   contour = args[1]
   vals, _ = _getGreyscaleHistogram(image, contour)
   return np.argmax(vals)

def getGreyAreaPercent(args):
   if not isinstance(args[0], VmsImage) or \
      not isinstance(args[1], type(np.zeros(1))): 
      return -1

   image = args[0]
   contour = args[1]
   vals, bins = _getGreyscaleHistogram(image, contour)
   mask_gry = bins <= 80
   vals_gry = vals[mask_gry[:-1]]
   area = sum(np.diff(bins)*vals)

   # Percent area above 200 (lighter pixels)
   return sum(vals_gry) / area

def getGreyArea(args):
   if not isinstance(args[0], VmsImage) or \
      not isinstance(args[1], type(np.zeros(1))): 
      return -1

   image = args[0]
   contour = args[1]
   vals, bins = _getGreyscaleHistogram(image, contour)
   return sum(np.diff(bins)*vals)

def getHsvAreaPercent(args):
   if not isinstance(args[0], VmsImage) or \
      not isinstance(args[1], type(np.zeros(1))): 
      return -1

   image = args[0]
   contour = args[1]
   hist = _getHsvHistogram(image, contour)
   return np.sum(hist[10:50]) / np.sum(hist)

def getHsvMean(args):
   if not isinstance(args[0], VmsImage) or \
      not isinstance(args[1], type(np.zeros(1))): 
      return -1

   image = args[0]
   contour = args[1]
   hist = _getHsvHistogram(image, contour)
   return np.mean(hist)

def getHsvArea(args):
   if not isinstance(args[0], VmsImage) or \
      not isinstance(args[1], type(np.zeros(1))): 
      return -1

   image = args[0]
   contour = args[1]
   hist = _getHsvHistogram(image, contour)
   return np.sum(hist)

def getHsvPeak(args):
   if not isinstance(args[0], VmsImage) or \
      not isinstance(args[1], type(np.zeros(1))): 
      return -1

   image = args[0]
   contour = args[1]
   hist = _getHsvHistogram(image, contour)
   return np.max(hist)

def getShapeMomentX(args):
   if not isinstance(args[1], type(np.zeros(1))): 
      return -1

   contour = args[1]
   M = cv.moments(contour)
   if M['m00'] == 0: return 0
   return int(M['m10'] / M['m00'])

def getShapeMomentY(args):
   if not isinstance(args[1], type(np.zeros(1))): 
      return -1

   contour = args[1]
   M = cv.moments(contour)
   
   if M['m00'] == 0: return 0
   return int(M['m01'] / M['m00'])

def getShapeArea(args):
   if not isinstance(args[1], type(np.zeros(1))): 
      return -1

   contour = args[1]
   return cv.contourArea(contour)

def getShapePerimiter(args):
   if not isinstance(args[1], type(np.zeros(1))): 
      return -1

   contour = args[1]
   return cv.arcLength(contour, True)

def getShapeAspectRatio(args):
   if not isinstance(args[1], type(np.zeros(1))): 
      return -1

   contour = args[1]
   _, _, w, h = cv.boundingRect(contour)
   return float(w) / h

def getShapeExtent(args):
   if not isinstance(args[1], type(np.zeros(1))): 
      return -1

   contour = args[1]
   _, _, w, h = cv.boundingRect(contour)
   # proportion of area filled by bounding rectangle
   area_rect = w * h
   return float(getShapeArea(args)) / area_rect

def getShapeSolidity(args):
   if not isinstance(args[1], type(np.zeros(1))): 
      return -1

   contour = args[1]
   hull = cv.convexHull(contour)
   return float(getShapeArea(args)) / cv.contourArea(hull)

def getShapeAngle(args):
   if not isinstance(args[1], type(np.zeros(1))): 
      return -1

   contour = args[1]
   #fit elipse, major minor axis, orientation angle
   (_, _), (_, _), angle = cv.fitEllipse(contour)
   return angle

## Returns dataframe of features relating to image
def getVmsFeatures(image):
   if isinstance(image, VmsImage):
      features = pd.DataFrame()
      contours = image.getContours()

      for contour in contours:
         feat_row = _getContourFeatures(image, contour)
         features = features.append(feat_row, ignore_index=True, sort=False)
   return features.dropna(how='any')

def getVmsDirFeatures(dir_in, segmentation_method = SegmentationTypeEnum.GREY):
   # ensure they are paths
   dir_in = os.path.abspath(dir_in)
   # dir_out = os.path.abspath(dir_out)

   if not os.path.exists(dir_in):
      return None

   os.chdir(dir_in)
   files = os.listdir()

   if len(files) < 1:
      return None

   features = pd.DataFrame()

   for file in files:
      image = VmsImage(file)
      image.setSegmentationMethod(segmentation_method)
      contour_features = getVmsFeatures(image)
      features = features.append(contour_features, 
                                 ignore_index=True,
                                 sort=False)
   return features

# Feature name with Callback function
feature_set = { 
   # 'gry_peak' : getGreyPeak,
   # 'gry_area' : getGreyArea,
   'gry_area_percent' : getGreyAreaPercent,
   'gry_mean' : getGreyMean,
   'hsv_area_percent' : getHsvAreaPercent,
   'hsv_mean' : getHsvMean,
   # 'hsv_peak' : getHsvPeak,
   # 'hsv_area' : getHsvArea,
   'shape_area' : getShapeArea,
   # 'shape_angle' : getShapeAngle,
   # 'shape_solidity' : getShapeSolidity,
   'shape_extent' : getShapeExtent,
   # 'shape_aspectRatio' : getShapeAspectRatio,
   # 'shape_perimeter' : getShapePerimiter,
   # 'shape_centroidX' : getShapeMomentX,
   'shape_centroidY' : getShapeMomentY,
}

if __name__ == '__main__':
   # Add Pickel Calls here pd.DataFrameToPickle
   dir_base = os.path.abspath("C:/git/python-examples/vms_detection/images")
   dir_out = os.path.abspath("C:/git/python-examples/vms_detection/out/")
   features = getVmsDirFeatures(dir_base)

   f_out = os.path.join(dir_out,"features.csv")
   features.to_csv(f_out, sep='\t', index=False)