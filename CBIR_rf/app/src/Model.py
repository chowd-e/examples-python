# PixInfo.py
# Program to start evaluating an image in python

import os, sys, glob
import numpy as np
from PIL import Image
from cmath import inf

from Globals import Mode

# Pixel Info class.
class PixInfo:

   # Constructor.
   def __init__(self, parent):
    
      self.parent = parent
      self.imageList = []
      self.photoList = []
      self.isRelevant = []
      self.xmax = 0
      self.ymax = 0
      self.colorCode = []
      self.intenCode = []
      self.comboCodes = []

      # set path based on exe or not
      if getattr(sys, 'frozen', False):
         app_path = os.path.dirname(sys.executable)
      else:
         app_path = os.path.dirname(os.path.abspath(__file__))

      self.imDir = os.path.join(app_path, 'images')
      print('Searching for images..\n' + str(self.imDir))
        
      # Add each image (for evaluation) into a list, 
      # and a Photo from the image (for the GUI) in a list.
      for infile in glob.glob(str(self.imDir) + "/*.jpg"):            
         im = Image.open(infile)
         self.isRelevant.append(False)
         # Resize the image for thumbnails.
         imSize = im.size
         x = imSize[0] // 4
         y = imSize[1] // 4
         imResize = im.resize((x, y), Image.ANTIALIAS)
         imResize.filename = im.filename
         
         # extract feature information from image
         pixList = list(im.getdata())
         CcBins, InBins = self.encode(pixList)

         pixCount = imSize[0] * imSize[1]
         self.colorCode.append(CcBins / pixCount)
         self.intenCode.append(InBins / pixCount)
         
         # Find the max height and width of the set of pics.
         if x > self.xmax:
            self.xmax = x
         if y > self.ymax:
            self.ymax = y
         
         # Add the images to the lists.
         self.imageList.append(im)
         self.photoList.append(imResize)

      self.comboCodes = self.combine(self.colorCode, self.intenCode)
      self.set_weights()
      
   # Bin function returns an array of bins for each 
   # image, both Intensity and Color-Code methods.
   def encode(self, pixlist):
      CcBins = np.zeros(64)
      InBins = np.zeros(25)

      # iterate through each pixel in the image
      for pix in pixlist:
         # Intensity conversion
         val = self.get_bins_intensity(pix)
         InBins[val] += 1
      
         # Color Conversion
         # convert integer to binary number, then string, index string for 
         idx = self.get_bins_color(pix)
         CcBins[idx] += 1
      
      # Return the list of binary digits, one digit for each
      # pixel.
      return CcBins, InBins
   
   def combine(self, list1, list2):
      if len(list1) != len(list2):
         return None

      result = []
      # Need to reassign memory for appended array
      for i in range(len(list2)):
         result.append(np.append(list1[i], list2[i]))
      return self.normalize(result)
       

   # normalize a list of arrays using Gaussian Normailization columnwise
   def normalize(self, list):
      mean = np.mean(list, axis = 0)
      std = np.std(list, axis = 0, ddof = 1)

      # replace zeroes with 1 so std doesn't make a difference
      std = np.where(std == 0, 1, std)

      list = np.subtract(list, mean)
      list = np.divide(list, std)
      return list

   def clear_checked(self):
      for i in range(len(self.isRelevant)):
         self.isRelevant[i] = False
      return 0

   # convert RGB value to intensity value and return appropriate bin 
   # for histogram
   def get_bins_intensity(self, pix):
      # Intensity conversion
      result = (0.299 * pix[0]) + (0.587 * pix[1]) + (0.114 * pix[2])
      result = int(result // 10)
      
      if result > 24:
         result = 24
      return result

   # Convert RGB value to 6-bit [0,64) value for histogram index storage
   def get_bins_color(self, pix):
      # Mask of 0b11000000 for bitwise AND operatiosn
      mask = 192

      # Bitwise operation to isolate two bits fo interest
      # Right shift to position for OR combination to 6-bit
      msb = (pix[0] & mask) >> 2
      mid = (pix[1] & mask) >> 4
      lsb = (pix[2] & mask) >> 6

      # OR all three sections to combine to 6-bit integer
      result = (msb | mid | lsb )
      return result

   # Get list of indices for relevant / chekced images
   def get_relevant_idx(self):
      idx = []
      for i in range(len(self.photoList)):
         if self.isRelevant[i]:
            idx.append(i)
      return idx

   # getters:
   def get_imageList(self):
      return self.imageList
   
   def get_photoList(self):
      return self.photoList
   
   def get_xmax(self):
      return self.xmax
   
   def get_ymax(self):
      return self.ymax

   def get_imDir(self):
      return self.imDir

   def get_checked(self):
      return self.isRelevant

   def get_codes(self, mode):
      if mode == Mode.COLOR:
         return self.colorCode
      elif mode == Mode.INTENSITY:
         return self.intenCode
      elif mode == Mode.COLOR_INTENSITY:
         return self.comboCodes
      else:
         return None
      
   def get_weights(self):
      return self.weights
   
   # setters:
   def set_checked(self, relevant):
      if len(relevant) != len(self.isRelevant):
         return -1
      else:
         self.isRelevant = np.where(relevant != -1, relevant, self.isRelevant)
      self.isRelevant = [bool(x) for x in self.isRelevant]
      return 0

   # Must be called after comboCodes has been filled
   def set_weights(self, with_relevant = False):

      if len(self.comboCodes[0]) == 0:
         return -1
      elif with_relevant == False:
         self.weights = np.ones_like(self.comboCodes[0])
         self.weights /= len(self.comboCodes[0])
      else:
         # Calculate co-variance among relevant image features
         # subset codes based on image subset #
         relevant = self.get_relevant_idx()
         if len(relevant) <= 1:
            return -1

         codes = self.get_codes(Mode.COLOR_INTENSITY)

         # subset for holding codes from relevant images
         subset = []
         for idx in relevant:
            subset.append(codes[idx])

         # STD Across columns #
         std = np.std(subset, axis = 0, ddof = 1)
         mean = np.mean(subset, axis = 0)
         std = np.where(std == 0, inf, std)
         # UPdate weights
         for i in range(len(std)):
            # Special case of standard dev = 0
            if std[i] == inf:
               if mean[i] == 0:
                  self.weights[i] = 1 / (0.5 * min(std))
               else:
                  # Set this weight to zero, remove from std
                  self.weights[i] = 0
            else:
               self.weights[i] = 1 / std[i]

         # Normalize Weights
         self.weights /= sum(self.weights) 
      return 0

if __name__ == '__main__':
   PixInfo(None)