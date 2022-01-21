# PixInfo.py
# Program to start evaluating an image in python

import os, sys, glob
from PIL import Image
from pathlib import Path

# Pixel Info class.
class PixInfo:

   # Constructor.
   def __init__(self, parent):
    
      self.parent = parent
      self.imageList = []
      self.photoList = []
      self.xmax = 0
      self.ymax = 0
      self.colorCode = []
      self.intenCode = []

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
         
         # Resize the image for thumbnails.
         imSize = im.size
         x = imSize[0] // 4
         y = imSize[1] // 4
         imResize = im.resize((x, y), Image.ANTIALIAS)
         imResize.filename = im.filename
         
         # extract feature information from image
         pixList = list(im.getdata())
         CcBins, InBins = self.encode(pixList)
         self.colorCode.append(CcBins)
         self.intenCode.append(InBins)
         
         # Find the max height and width of the set of pics.
         if x > self.xmax:
            self.xmax = x
         if y > self.ymax:
            self.ymax = y
         
         # Add the images to the lists.
         self.imageList.append(im)
         self.photoList.append(imResize)

   # Bin function returns an array of bins for each 
   # image, both Intensity and Color-Code methods.
   def encode(self, pixlist):
      CcBins = [0]*64
      InBins = [0]*25

      # iterate through each pixel in the image
      for pix in pixlist:
         # Intensity conversion
         val = self.getIntensityBin(pix)
         InBins[val] += 1
      
         # Color Conversion
         # convert integer to binary number, then string, index string for 
         idx = self.getColorBin(pix)
         CcBins[idx] += 1
      
      # Return the list of binary digits, one digit for each
      # pixel.
      return CcBins, InBins
    
   # convert RGB value to intensity value and return appropriate bin 
   # for histogram
   def getIntensityBin(self, pix):
      # Intensity conversion
      result = (0.299 * pix[0]) + (0.587 * pix[1]) + (0.114 * pix[2])
      result = int(result // 10)
      
      if result > 24:
         result = 24
      return result

   # Convert RGB value to 6-bit [0,64) value for histogram index storage
   def getColorBin(self, pix):
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

   # getters:
   def get_imageList(self):
      return self.imageList
   
   def get_photoList(self):
      return self.photoList
   
   def get_xmax(self):
      return self.xmax
   
   def get_ymax(self):
      return self.ymax
   
   def get_colorCode(self):
      return self.colorCode
      
   def get_intenCode(self):
      return self.intenCode

if __name__ == '__main__':
   PixInfo(None)