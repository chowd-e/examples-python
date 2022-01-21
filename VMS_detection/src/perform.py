import pickle
import segmentation, extraction
import cv2 as cv

def exportModel(model, filename = 'model.pkl'):
   try:
      # validate file extention
      if ~filename.endswith('.pkl'):
         filename += ".pkl"

      with open(filename, 'wb') as f:
         pickle.dump(model, f)
   except:
      return -1

   return 0

def loadModel(path):
   try:
      with open(path, 'rb') as f:
         model = pickle.load(f)
   except:
      return None

   return model

def getContours(IM, method = None):
   if method == 'color':
      contours = segmentation.getContourColor(IM)
   elif method == 'threshold':
      contours = segmentation.getContourThresh(IM)
   else:
      contours = [segmentation.getContourColor(IM) 
                  + segmentation.getContourThresh(IM)]

   return contours

   # Container for feature data
   featureRows = []

   for cnt in contours:
      # Subset image based on contour bounds
      x, y, w, h = cv.boundingRect(cnt)
      img_sub = IM[y:y + h, x:x + w]      

      # Extract Features
      greyRow = extraction.histGrey(img_sub)
      colorRow = extraction.histColor(img_sub) 
      shapeRow = extraction.shape(cnt)

      featureRow = ([method] + greyRow + colorRow + shapeRow )

      # cv.destroyAllWindows() 
   return featureRows



