import os, csv
import segmentation, extraction
# import modelEvaluation as me
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def isVMS():
   val = bool(input("Input char if Contour contains VMS, Enter to continue:"))
   if val:
      return [1]
   else:
      return [0]

def main():
   # prompt dir
   dir_rd = 'C:/git/UW/581-ml/proj/data/test'
   dir_wr = 'C:/git/UW/581-ml/proj/data/labels'
   # dir_rd = askdirectory(title = 'Select Folder to load Files')
   os.chdir(dir_rd)
   data = []

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
         greyRow = extraction.histGrey(img_sub)
         colorRow = extraction.histColor(img_sub) 
         shapeRow = extraction.shape(cnt)
         data.append([0] + greyRow + colorRow + shapeRow )
         # cv.destroyAllWindows() 
      
      for cnt in cntColor:
         # Subset image based on contour bounds
         x, y, w, h = cv.boundingRect(cnt)
         img_sub = img[y:y + h, x:x + w]      
         # cv.imshow("", img_sub)

         greyRow = extraction.histGrey(img_sub)
         colorRow = extraction.histColor(img_sub) 
         shapeRow = extraction.shape(cnt)
         data.append([1] + greyRow + colorRow + shapeRow ) 
         # cv.destroyAllWindows()

def writeDelimitedData(path, filename, data):
   path_og = os.getcwd()
   os.chdir(path)
   with open(filename, 'w', newline="") as f :
      write = csv.writer(f)
      write.writerows(data)

   # set to original path
   os.chdir(path_og)

if __name__ == "__main__":
   main()
