import sys, os, pickle
import pandas as pd
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from VmsImage import SegmentationTypeEnum, VmsImage
import FeatureExtraction as feat
import ModelEvaluation as eval 

def main():
   target = "Variable Message Sign"
   model = Model()
   path_in = os.path.abspath(
      "C:/git/python-examples/VMS_detection/src/images/001.jpg")
   model.loadImage(path_in)
   # model.display_image()
   isVMS = model.getPrediction()

   if(isVMS):
      print("This is a %s" % target)
   else:
      print("This is not a %s" % target)

class Model:

   # get relative path to subdir within launching instance
   def __init__(self, base_path="") -> None:
      # Requires set of pre-trained models
      self.models_ = {
         'DecisionTree' : "models/DecisionTree.pkl",
         'LogisticRegression' : "models/LogisticRegression.pkl",
         'NaiveBayes' : "models/NaiveBayes.pkl",
         'RandomForest' : "models/RandomForest.pkl",
         'XGBoost': "models/XGBoost.pkl",
      }
      
      if base_path == "":
         base_path = self.getAbsolutePath()

      self.basePath_ = os.path.abspath(base_path)
      self.loadModel()
      self.image_ = None
      self.features_ = pd.DataFrame()

   def setViewer(self, viewer):
      self.viewer = viewer
      return 0

   def getModels(self):
      return [*self.models_]

   def getAbsolutePath(self, subdir = ""):
      if getattr(sys, 'frozen', False):
         app_path = os.path.dirname(sys.executable)
      else:
         app_path = os.path.dirname(os.path.abspath(__file__))

      if subdir == "":
         return app_path
      else:
         return os.path.join(app_path, subdir)

   def loadImage(self, path):
      self.image_ = VmsImage(path)
      self.image_.setSegmentationMethod(SegmentationTypeEnum.COLOR)
      if self.image_.getPath() == None:
         print("Error Loading File - Check path")
         return None
      self.features_ = pd.DataFrame()
      return 0

   def loadModel(self, selection = 'DecisionTree'):
      if selection in self.models_:
         path = os.path.join(self.basePath_,self.models_[selection])
      else:
         return None

      try:
         with open(path, 'rb') as f:
            self.model_ = pickle.load(f)
         return self.model_
      except:
         self.model_ = None
         return None

   def getPrediction(self):
      if self.image_ == None:
         return None

      if self.features_.empty:
         self.features_ = feat.getVmsFeatures(self.image_)

      preds = self.model_.predict(self.features_)
      return preds
   
   def getContours(self):
      if(self.image_ == None):
         return
      return self.image_.getContours()

   def setSegmentationType(self, method):
      if isinstance(method, SegmentationTypeEnum):
         return self.image_.setSegmentationMethod(method)
      else:
         return None
      
   def getFeatures(self):
      try:
         if self.features_.empty:
            self.features_ = feat.getVmsFeatures(self.image_)
         
         out = pd.DataFrame(self.features_)
         # create subplots and populate
         fig = plt.figure()
         gs = gridspec.GridSpec(1,1)

         ax_table = fig.add_subplot(gs[0])
         eval.plotTable(out, 'Image Features', ax=ax_table)         
         plt.show()
         return fig
      except:
         return None

   def evaluateModel(self, path_source = ""):
      # Load data from source, where to get this from???
      data, _ = eval.loadFeatures(path_source)
      eval.showConfusionMatrix(data, self.model_)
      metrics = eval.computeMetrics(data, self.model_)
      eval.plotTable(metrics, 'Metrics Summary')
      plt.show()
      return 0

   def getPath(self):
      return self.image_.getPath()

if __name__ == '__main__':
   main()
