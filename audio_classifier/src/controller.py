import sys
from PyQt5.QtWidgets import QApplication

from view import Viewer
from model import Model

class Controller:
   
   def recv(self, cmd, args = []):
      if cmd in self.commands:
         return self.commands[cmd](args)
      else:
         print("Command not found: \"%s\"\n\nCommand List:" % cmd)
         print(*self.commands, sep ="\n")
         return -1

   def load_audio(self, path):
      return self.backend.load_audio(path)

   def play_audio(self,_):
      return self.backend.play_audio()

   def get_audio_details(self,_):
      return self.backend.get_features()

   def evaluate_model(self,source_data):
      # Get precision, recall, f1 and confusion matrix
      return self.backend.evaluate_model(source_data)

   def load_model(self, model_name):
      return self.backend.load_model(model_name)

   def get_prediction(self,_):
      return self.backend.get_prediction()

   def train_model(self, data = []):
      pass

   def __init__(self) -> None:
      self.commands = {
         'load' : self.load_audio,
         'play' : self.play_audio,
         'audio' : self.get_audio_details,
         'model' : self.evaluate_model,
         'predict':self.get_prediction,
         'load_model':self.load_model,
         'train' : self.train_model,
      }
      
      self.backend = Model()
      self.viewer = Viewer(self, self.backend)
      self.backend.set_viewer(self.viewer)

      

if __name__ == '__main__':
   app = QApplication(sys.argv)
   tmp = Controller()
   sys.exit(app.exec_())
