# This class is designed to extract a set of features from a given image, 
# features include greyscale, color, and shape features based on contours or 
# subsets of images
import os
import librosa
import pandas as pd
import numpy as np

from AudioSegment import AudioSegment
"""
This script is used to extract features from an AudioSegment
"""
def get_all_featureNames():
   return [*feature_set]

def process_feature_command(cmd, args = []):
   if cmd in feature_set:
      return feature_set[cmd](args)
   else:
      print("Feature not found: \"%s\"\n\nFEATURE LIST:" % cmd)
      print(*feature_set, sep ="\n")
      return -1

def get_feature_set(segment, featureNames = []):
   features = dict()
   # ALL FEATURES
   if featureNames == []:
      for feat in feature_set.keys():
         features[feat] = process_feature_command(feat, segment)
   # Select Features
   else:
      for feat in featureNames:
         features[feat] = process_feature_command(feat, segment)

   return features

# clip is a segmented audio signal (time, amplitude) in the time domain 
def get_energy(segment):   
   amp = segment.get_amplitude()

   if len(amp) <= 0:
      return -1

   return np.sum(np.power(amp, 2)) / len(amp)

def get_zero_crossing_rate(segment):
   out = librosa.feature.spectral.zero_crossings(y = segment.get_amplitude())
   return np.mean(out)

# Analyze shape features associated with the contour of interest
def get_bandwidth(segment, threshold_db = 3):
   # Get FFT
   freq, db = segment.get_xy_frequency()
   # set threshold of 3dB
   freq = freq[db > threshold_db]

   if len(freq) == 0:
      return 0 
   
   # max minus min of freqencies which are above threshold_dB
   bandwidth = max(freq) - min(freq)
   return float(bandwidth)

def get_spectral_contrast(segment):
   out = librosa.feature.spectral_contrast(y=segment.get_amplitude(),
                                     sr = segment.get_sample_rate()
   )
   return np.mean(out)

def get_RMS(segment):
   out = librosa.feature.rms(y=segment.get_amplitude())
   return np.mean(out)

def get_spectral_flatness(segment):
   out = librosa.feature.spectral_flatness(y=segment.get_amplitude())
   return np.mean(out)

def get_spectral_bandwidth(segment):
   out = librosa.feature.spectral_bandwidth(y =segment.get_amplitude(),
                                            sr= segment.get_sample_rate()
                     )
   return np.mean(out)

# these are application specific frequency bands and the power associated with
# those specific bands
def get_energy_distribution(segment, threshold_hz = 4000):
   freq, db = segment.get_xy_frequency()
   mask = freq <= threshold_hz

   mag_low = db[mask]
   mag_low = np.sum(np.power(mag_low, 2)) / len(db)

   mag_high = db[~mask] 
   mag_high = np.sum(np.power(mag_high, 2)) / len(db)

   ratio = mag_low / (mag_low + mag_high)
   return ratio

def get_silence_ratio(segment, threshold = 0.4):
   amplitude = abs(segment.get_amplitude())
   amplitude = segment.normalize(amplitude)
   total_samples = len(amplitude)
   n_samples = len(amplitude[amplitude > threshold])
   return n_samples / total_samples

def get_spectral_centroid(segment):
   hz, db = segment.get_xy_frequency()
   return np.divide(np.sum(np.multiply(hz, db)),np.sum(db))

def get_spectral_centroid(segment):
   out = librosa.feature.spectral_centroid(y = segment.get_amplitude(),
                                           sr = segment.get_sample_rate()
   )
   return out.mean()

feature_set = { 
   'Bandwidth' : get_bandwidth,
   'ZeroCrossingRate' : get_zero_crossing_rate,
   'SilenceRatio' : get_silence_ratio,
   'RMS' : get_RMS,
   'Energy' : get_energy,
   'EnergyDistribution' : get_energy_distribution,
   'SpectralCentroid' : get_spectral_centroid,
   'SpectralBandwidth' : get_spectral_bandwidth,
   'SpectralContrast' : get_spectral_contrast,
   'SpectralFlatness' : get_spectral_flatness,
}

def get_clip_features(audio, dt_ms = 60000):
   clips = audio.sub_sample(dt_ms)
   # features = pd.DataFrame(columns = get_all_featureNames())
   features = pd.DataFrame()
   for c in clips:
      # Don't subset clips shorter than 1 second
      if c.get_duration() < 1000:
         continue

      feat_row = get_feature_set(c)
      features = features.append(feat_row, ignore_index=True, sort=False)
      # print("Processed %s: %0.1f" % (str(f), c.get_duration() / 1000))
   return features.dropna(how='any')

def extract_features_from_dir(dir_in, dir_out):
   # ensure they are paths
   dir_in = os.path.abspath(dir_in)
   dir_out = os.path.abspath(dir_out)

   if not os.path.exists(dir_in):
      return None

   os.chdir(dir_in)
   files = os.listdir()

   if len(files) < 1:
      return None

   cols = get_all_featureNames()

   header = cols + ['isMusic']
   features = pd.DataFrame(columns = header)

   for f in files:
      path_abs = os.path.abspath(f)
      audio = AudioSegment(path_abs)
      
      clip_feat = get_clip_features(audio)

      features = features.append(clip_feat, 
                                 ignore_index=True,
                                 sort=False)

   f_out = os.path.join(dir_out,"features.csv")
   features.to_csv(f_out, sep='\t', index=False)

   return features

if __name__ == '__main__':
   # Add Pickel Calls here pd.DataFrameToPickle
   dir_base = os.path.dirname(
      "C:/git/proj/audio_ml/supplied_audio/"
   )

   dir_out = os.path.dirname("C:/git/proj/audio_ml/out/")
   
   extract_features_from_dir(dir_base, dir_out)