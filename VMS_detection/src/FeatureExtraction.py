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
This script is used to extract features from a VmsImage
"""
def get_all_featureNames():
   return [*feature_set]

def process_feature_command(cmd, args = []):
   if cmd in feature_set and isinstance(args[0], VmsImage):
      return feature_set[cmd](args)
   else:
      print("Feature not found: \"%s\"\n\nFEATURE LIST:" % cmd)
      print(*feature_set, sep ="\n")
      return -1

def getFeatureSet(image, contour, featureNames = []):
   features = dict()
   # ALL FEATURES
   if featureNames == []:
      for feat in feature_set.keys():
         features[feat] = process_feature_command(feat, [image, contour])
   # Select Features
   else:
      for feat in featureNames:
         features[feat] = process_feature_command(feat, [image, contour])
         
   # features["Method"] = image.getSegmentationMethod()
   return features   

def _getGreyscaleHistogram(image, contour):
   #  subset image based on contour bounds to retain color
   image = image.getImageSubset(contour)
   gry = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
   gry = cv.GaussianBlur(gry, (19, 19), 0)

   vals, bins, _ = plt.hist(gry.ravel(), bins=255, range=(0, 255), fc='k', ec='k')
   return vals, bins

def getGreyMean(image, contour):
   vals, _ = _getGreyscaleHistogram(image, contour)
   return np.mean(vals)

def getGreyPeak(image, contour):
   vals, _ = _getGreyscaleHistogram(image, contour)
   return np.argmax(vals)

def getGreyAreaPercent(image, contour):
   vals, bins = _getGreyscaleHistogram(image, contour)
   mask_gry = bins <= 80
   vals_gry = vals[mask_gry[:-1]]
   area = sum(np.diff(bins)*vals)

   # Percent area above 200 (lighter pixels)
   return sum(vals_gry) / area

def getGreyArea(image, contour):
   vals, bins = _getGreyscaleHistogram(image, contour)
   return sum(np.diff(bins)*vals)

def _getHsvHistogram(image, contour):
   image = image.getImageSubset(contour)
   hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
   hsv = cv.GaussianBlur(hsv, (19,19), 0)

   bins = 180
   hist = cv.calcHist([hsv], [0], None, [bins], [0, bins])
   return hist 

def getHsvAreaPercent(image, contour):
   hist = _getHsvHistogram(image, contour)
   area = np.sum(hist)
   return np.sum(hist[10:50]) / area

def getHsvMean(image, contour):
   hist = _getHsvHistogram(image, contour)
   return np.mean(hist)


def getHsvArea(image, contour):
   hist = _getHsvHistogram(image, contour)
   return np.sum(hist)

def getHsvPeak(image, contour):
   hist = _getHsvHistogram(image, contour)
   return np.max(hist)

def getShapeMomentX(_,contour):
   M = cv.moments(contour)
   return int(M['m10'] / M['m00'])

def getShapeMomentY(_,contour):
   M = cv.moments(contour)
   return int(M['m01'] / M['m00'])

def getShapeArea(_,contour):
   return cv.contourArea(contour)

def getShapePerimiter(_,contour):
   return cv.arcLength(contour, True)

def getShapeAspectRatio(_,contour):
   _, _, w, h = cv.boundingRect(contour)
   return float(w) / h

def getShapeExtent(_,contour):
   _, _, w, h = cv.boundingRect(contour)
   # proportion of area filled by bounding rectangle
   area_rect = w * h
   return float(getShapeArea(None, contour)) / area_rect

def getShapeSolidity(_,contour):
   hull = cv.convexHull(contour)
   return float(getShapeArea(None, contour)) / cv.contourArea(hull)

def getShapeAngle(_,contour):
   #fit elipse, major minor axis, orientation angle
   (_, _), (_, _), angle = cv.fitEllipse(contour)
   return angle

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

## Returns dataframe of features relating to image
def getVmsFeatures(image):
   if isinstance(image, VmsImage):
      features = pd.DataFrame()
      contours = image.getContours()

      for contour in contours:
         feat_row = getFeatureSet(image, contour)
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

if __name__ == '__main__':
   # Add Pickel Calls here pd.DataFrameToPickle
   dir_base = os.path.abspath("C:/git/python-examples/vms_detection/images")
   dir_out = os.path.abspath("C:/git/python-examples/vms_detection/out/")
   features = getVmsDirFeatures(dir_base)

   f_out = os.path.join(dir_out,"features.csv")
   features.to_csv(f_out, sep='\t', index=False)