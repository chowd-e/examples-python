import os
import librosa
import numpy as np
from scipy import fft
from sklearn.preprocessing import minmax_scale

class AudioSegment:
   
   def __init__(self, *args) -> None:
      
      # Single paramter passed - load from file
      if len(args) == 1:
         self.load_from_file(args[0])
      # Pass two arguments, 
      elif len(args) == 3:
         self.load_from_segment(args[0], args[1], args[2])
      else:
         print('Invalid Constructor')

   # Load audio from existing file (import full audio clip)
   def load_from_file(self, path, sample_rate = 22050):
      
      if not os.path.exists(path):
         return None
      self.path = os.path.abspath(path)
      self.amplitude, self.sample_rate = librosa.load(path, sr = sample_rate)
      self.preprocessing()

      self.duration_ms = 1000 * len(self.amplitude) / self.sample_rate
      self.time = np.arange(0, 
                             self.duration_ms, 
                             1000 / self.sample_rate)
      self._set_frequency_from_amplitude()

   # initialize an AudioSegement from existing AudioSegment given start/stop   
   def load_from_segment(self, full_segment, start_ms, stop_ms):
      self.path = full_segment.get_path()
      self.sample_rate = full_segment.get_sample_rate()
      
      # Convert time to index
      start_idx = int(start_ms * self.sample_rate / 1000)
      stop_idx = int(stop_ms * self.sample_rate / 1000)

      self.amplitude = full_segment.get_amplitude()[start_idx : stop_idx]
      self.duration_ms = stop_ms - start_ms
      self.time = full_segment.get_time()[start_idx : stop_idx]
      self._set_frequency_from_amplitude()

   def preprocessing(self):
      # self.amplitude = self.normalize(self.amplitude)
      # emphasize high frequenc components in speech
      self.amplitude = librosa.effects.preemphasis(self.amplitude)
      return 0

   # Segment raw audio into a series of clips with length of 10s (or custom 
   # defined) starting from skip_ms
   def sub_sample(self, time_step_ms = 30000, skip_ms = 0):

      if self.duration_ms <= time_step_ms or time_step_ms == 0:
         return [self]

      # duration will be length in milliseconds
      clips = []
      start_ms = skip_ms # start at beggining of file
      while start_ms + time_step_ms < self.duration_ms:
         
         stop_ms = start_ms + time_step_ms
         clips.append(AudioSegment(self, start_ms, stop_ms))
         start_ms = stop_ms
      # Append final clip
      clips.append(AudioSegment(self,start_ms,self.duration_ms))
      return clips

   """
   Utilies
   """
   def db_to_magnitude(self, db):
      return np.power(10, (db / 20))

   def magnitude_to_db(self, magnitude):
      return 20 * np.log10(abs(magnitude))

   def normalize(self, x, axis = 0):
         return minmax_scale(x, axis = axis)
   """
   SETTERS
   """
   def _set_frequency_from_amplitude(self):
      magnitude = fft.fft(self.amplitude)
      # make sure to divide dt by 1000 for seconds -> Hz instead of ms -> mHz
      hz = fft.fftfreq(len(self.time), (self.time[1] - self.time[0]) / 1000)
      # Limit to postive values
      
      hz  = hz[hz >= 0]
      end = min(len(hz), len(magnitude))
      magnitude = magnitude[:end]
      db = self.magnitude_to_db(magnitude)
      # convert amplitude to db (not sure if I want this)

      self.frequency = [hz, db]
      return 0

   """
   GETTERS
   """
   def get_path(self):
      return self.path

   def get_xy_time(self):
      return [self.time, self.amplitude]

   def get_amplitude(self):
      return self.amplitude

   def get_time(self):
      return self.time

   def get_xy_frequency(self):
      return self.frequency

   def get_duration(self):
      return self.duration_ms

   def get_sample_rate(self):
      return self.sample_rate

if __name__ == "__main__":
   dir_base = "C:/Users/chase/..main/learn/UW/MS/CSS 584 - Multimedia Database Systems/assignments/04 - ML_audio/assignment4_supportFiles/audio/all"
   
   if os.path.exists(dir_base):     
      os.chdir(dir_base)
      files = os.listdir()

      for f in files:
         path_abs = os.path.abspath(f)
         audio = AudioSegment(path_abs)
