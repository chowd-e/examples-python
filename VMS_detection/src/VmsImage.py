import enum
import os
import Segmentation
import cv2 as cv

class SegmentationTypeEnum(enum.Enum):
   NONE = 0
   GREY = 1
   COLOR = 2
   ALL = 3

class VmsImage:
   
   def __init__(self, *args) -> None:
      # Single paramter passed - load from file
      if len(args) == 1:
         self.loadFromFile(args[0])
         # What other constructors do I need?
      else:
         print('Invalid Constructor: Usage: \nVmsImage(path) ')

   # Load image from existing file
   def loadFromFile(self, path):
      # Default to greyscale
      self.method_ = SegmentationTypeEnum.GREY
      # Path to image
      self.path_ = None
      # PIL Image store
      self.image_ = None
      # List of contours from segmented image
      self.contours_ = None

      if not os.path.exists(path):
         return None
      self.path_ = os.path.abspath(path)
      # load image and get contours
      self.image_ = cv.imread(self.path_)


   """
   Utilies
   """
   def getImageSubset(self, contour):
      x, y, w, h = cv.boundingRect(contour)
      return self.image_[y:y + h, x:x + w]

   """
   SETTERS
   """
   def setSegmentationMethod(self, method):
      if isinstance(method, SegmentationTypeEnum):
         if self.method_ != method:
            self.method_ = method
            # Type has changed, wipe contours
            self.contours_ = None
            return True
      return False

   def setPath(self, path):
      self.loadFromFile(path)

   """
   GETTERS
   """

   def getPath(self):
      return self.path_

   def getContours(self):

      # See if contours have been loaded, otherwise load them
      if self.contours_ == None:
         if self.method_ == SegmentationTypeEnum.GREY:
            self.contours_ = Segmentation.greyGetContour(self.image_)
         elif(self.method_ == SegmentationTypeEnum.COLOR):
            self.contours_ = Segmentation.colorGetContour(self.image_)
         elif(self.method_ == SegmentationTypeEnum.ALL):
            self.contours_ = Segmentation.getContours(self.image_)
         else:
            return None
      
      return self.contours_

   def getSegmentationMethod(self):
      return self.method_

if __name__ == "__main__":
   dir_base = "C:/git/python-examples/VMS_detection/images"
   
   if os.path.exists(dir_base):     
      os.chdir(dir_base)
      files = os.listdir()

      for f in files:
         path_abs = os.path.abspath(f)
         image = VmsImage(path_abs)
         image.setSegmentationMethod(SegmentationTypeEnum.ALL)
         cnt = image.getContours()
         print("contours")
