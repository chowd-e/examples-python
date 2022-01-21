# This class is for processing all of the images under inspection, it will 
# return a csv file with all features extracted and manually classified

import sys, os, csv
import segmentation, extraction
# import modelEvaluation as me
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
   # prompt dir
   dir_rd = getAbsolutePath(subdir='images')
   dir_wr = getAbsolutePath(subdir = "outputs")
   os.chdir(dir_rd)
   data = []
   label = 0

   for filename in os.listdir(dir_rd) :
      img = cv.imread(filename)
      
      cntColor = segmentation.getContourColor(img)
      cntThresh = segmentation.getContourThresh(img)
      # cntAll = segmentation.getContours(img)
      
      # Analyze Thresholds
      for cnt in cntThresh:
         # Subset image based on contour bounds
         x, y, w, h = cv.boundingRect(cnt)
         img_sub = img[y:y + h, x:x + w]      
         # cv.imshow("", img_sub)
         # label = segmentation.labelContour()
         greyRow = extraction.histGrey(img_sub)
         colorRow = extraction.histColor(img_sub) 
         shapeRow = extraction.shape(cnt)
         data.append([0] + greyRow + colorRow + shapeRow + label)
         # cv.destroyAllWindows() 
      
      for cnt in cntColor:
         # Subset image based on contour bounds
         x, y, w, h = cv.boundingRect(cnt)
         img_sub = img[y:y + h, x:x + w]      
         # cv.imshow("", img_sub)
         # label = segmentation.labelContour()
         greyRow = extraction.histGrey(img_sub)
         colorRow = extraction.histColor(img_sub) 
         shapeRow = extraction.shape(cnt)
         data.append([1] + greyRow + colorRow + shapeRow + label) 
         # cv.destroyAllWindows()

   writeDelimitedData(dir_wr, 'VMS_dataset.csv', data=data)

def writeDelimitedData(path, filename, data):
   path_og = os.getcwd()
   os.chdir(path)
   with open(filename, 'w', newline="") as f :
      write = csv.writer(f)
      write.writerows(data)
   # set to original path
   os.chdir(path_og)

# get absolute path to subdir within launching instance
def getAbsolutePath(subdir = ""):
   if getattr(sys, 'frozen', False):
      app_path = os.path.dirname(sys.executable)
   else:
      app_path = os.path.dirname(os.path.abspath(__file__))

   if subdir == "":
      return app_path
   else:
      return os.path.join(app_path, subdir)

if __name__ == "__main__":
   main()
