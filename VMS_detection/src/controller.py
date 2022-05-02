import sys
from PyQt5.QtWidgets import QApplication

from view import Viewer
from model import Model

class Controller:
   
   def __init__(self) -> None:
      self.commands = {
         'load' : self.loadImage,
         'draw' : self.drawContours,
         'details' : self.getDetails,
         'evaluate' : self.evaluateModel,
         'predict':self.getPrediction,
         'load_model':self.loadModel,
         'train' : self.trainModel,
      }
      
      self.model_ = Model()
      self.viewer_ = Viewer(self, self.model_)
      self.model_.setViewer(self.viewer_)
      
   def recv(self, cmd, args = []):
      if cmd in self.commands:
         return self.commands[cmd](args)
      else:
         print("Command not found: \"%s\"\n\nCommand List:" % cmd)
         print(*self.commands, sep ="\n")
         return -1

   def loadImage(self, path):
      return self.model_.loadImage(path)

   def display_image(self,_):
      return self.viewer_.displayImage()

   def drawContours(self):
      return self.model_.getContours()

   def getDetails(self,_):
      return self.model_.getFeatures()

   def evaluateModel(self,source_data):
      # Get precision, recall, f1 and confusion matrix
      return self.model_.evaluateModel(source_data)

   def loadModel(self, model_name):
      return self.model_.loadModel(model_name)

   def getPrediction(self,_):
      return self.model_.getPrediction()

   def trainModel(self, data = []):
      pass

if __name__ == '__main__':
   app = QApplication(sys.argv)
   tmp = Controller()
   sys.exit(app.exec_())
