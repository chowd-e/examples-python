import sys, os, pickle
import pandas as pd
import numpy as np
from playsound import playsound

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from AudioSegment import AudioSegment
import FeatureExtraction as feat
import ModelEvaluation as eval 

def main():
   backend = Model()
   path_in = os.path.abspath("put path here")
   backend.load_audio(path_in)
   backend.play_audio()
   isMusic = backend.get_prediction()

   if(isMusic):
      print("This is a clip of Music")
   else:
      print("This is not music")

class Model:

   # get relative path to subdir within launching instance
   def __init__(self, base_path="") -> None:
      self.models = {
         'Decision_Tree' : "models/DecisionTree.pkl",
         'Logistic_Regression' : "models/LogisticRegression.pkl",
         'Naive_Bayes' : "models/NaiveBayes.pkl",
         'Random_Forest' : "models/RandomForest.pkl",
         'XGBoost': "models/XGBoost.pkl",
      }
      
      if base_path == "":
         base_path = self.getAbsolutePath()

      self.path_base = os.path.abspath(base_path)
      self.load_model()
      self.audio = None

   def set_viewer(self, viewer):
      self.viewer = viewer
      return 0

   def get_models(self):
      return [*self.models]

   def getAbsolutePath(self, subdir = ""):
      if getattr(sys, 'frozen', False):
         app_path = os.path.dirname(sys.executable)
      else:
         app_path = os.path.dirname(os.path.abspath(__file__))

      if subdir == "":
         return app_path
      else:
         return os.path.join(app_path, subdir)

   def load_audio(self, path):
      self.audio = AudioSegment(path)
      if self.audio == None:
         print("Error Loading File - Check path")
         return None

      self.features = feat.get_clip_features(self.audio)
      return 0

   def load_model(self, selection = 'Decision_Tree'):
      if selection in self.models:
         path = os.path.join(self.path_base,self.models[selection])
      else:
         return None

      try:
         with open(path, 'rb') as f:
            self.model = pickle.load(f)
         return self.model
      except:
         self.model = None
         return None

   def get_prediction(self):
      if self.audio == None or self.features.empty:
         return None

      preds = self.model.predict(self.features)
      pred = round(np.mean(preds))  
      return pred

   def play_audio(self, path = None):
      if path != None:
         self.load_audio(path)

      if self.audio == None:
         return None
         
      playsound(self.audio.get_path())
      return 0

   def get_time(self):
      return self.audio.get_time()

   def get_freq(self):
      return self.audio.get_xy_frequency()

   def get_amp(self):
      return self.audio.get_amplitude()

   def get_waveform(self):
      return [self.get_time(), self.get_amp()]
   
   
   def get_features(self):
      try:
         out = pd.DataFrame(self.features)
         # create subplots and populate
         fig = plt.figure()
         gs = gridspec.GridSpec(4,1)

         ax_table = fig.add_subplot(gs[0])
         eval.plot_table(out, 'Audio Features', ax=ax_table)
         
         ax_amp = fig.add_subplot(gs[1])
         ax_amp.plot(self.get_time(),self.get_amp())
         ax_amp.set_title('Amplitude over time', fontweight='bold')
         ax_amp.set_xlabel('Time [ms]')
         ax_amp.set_ylabel('Amplitude')

         
         ax_freq = fig.add_subplot(gs[3])
         hz, db = self.get_freq()
         ax_freq.plot(hz, db)
         ax_freq.set_title('Frequency Distribution', fontweight='bold')
         ax_freq.set_xlabel('Frequency [hz]')
         ax_freq.set_ylabel('Magnitude [db]')
         
         plt.show()
         return fig
      except:
         return None

   def evaluate_model(self, path_source = ""):
      # Load data from source, where to get this from???
      data, _ = eval.load_features(path_source)
      eval.display_confusion_mat(data, self.model)
      metrics = eval.evaluate_metrics_on_model(data, self.model)
      eval.plot_table(metrics, 'Metrics Summary')
      plt.show()
      return 0

   def get_path(self):
      return self.audio.get_path()

if __name__ == '__main__':
   main()
