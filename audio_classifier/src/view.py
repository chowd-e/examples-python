import os
from PyQt5.QtWidgets import QLabel, QPushButton, QComboBox, QHBoxLayout, QVBoxLayout, QMainWindow, QWidget, QMessageBox
from PyQt5.QtWidgets import QFileDialog as qfd
from PyQt5.QtCore import Qt

class Viewer(QMainWindow):
   # display search bar for loading file name
   
   def __init__(self, controller, backend) -> None:
      super().__init__()
      # Main Widet
      self.main = QWidget(self)
      self.setCentralWidget(self.main)
      
      self.backend = backend
      self.control = controller

      self.build()
      
   def send(self, cmd, args=[]):
      return self.control.recv(cmd,args)

   def build(self):
      self.title = 'CSS 584 - Audio Classification'
      self.left = 100
      self.top = 100
      self._width = 420
      self._height = 180

      # Button References
      self.init_ui()

   ##### BUTTON RESPONSES #####
   def predict(self):
      resp = self.send('predict')
      # Get key from value in dict
      
      if resp == None:
         self.dialog("Please load an audio file first")
         return 0
      elif resp == 0:
         val = "is Not Music"
      else:
         val = "is Music"
      self.dialog("Prediction: " + val)
      return 0
      
   def playback(self):
      self.disp("Playing file...")
      resp = self.send('play')

      if resp == None:
         self.dialog("Please load an audio file first")
      else:
         self.disp("Playback complete")
      return 0

   def load_audio(self):
      path = qfd.getOpenFileName(caption="Select .wav file",
                                        directory="audio")[0]
      
      if path == '':
         self.disp("Operation canceled, no audio file selected")
         return 0
      
      self.disp('Loading File...')
      resp = self.send('load', path)
      f = os.path.basename(path)

      if resp == 0:
         self.disp("New file loaded: " + str(f))
         self.update_fname()
      else:
         self.disp("Error loading Audio")
      return 0 

   def show_audio_details(self):
      resp = self.send('audio')
      if resp is not None:
         self.disp('Features retrieved')
      else:
         self.dialog('Please load an audio file first')
      # resp will be dataframe fo features, how to display
      return resp

   def show_model_details(self):
      self.disp("Select Source Data")
      source_data = qfd.getOpenFileName(caption="Select Source Data",
                                        directory="data")[0]

      if source_data == '':
         self.disp("Canceled, No data selected to evaluate model perfomance")
         return 0

      return self.send('model', source_data)

   def set_model(self):
      caller = self.sender()
      # Need to get new model with this text
      resp = self.send('load_model', caller.currentText())
      if resp is not None:
         self.disp("Updated Model: " + caller.currentText())
      else:
         self.dialog("Error loading Model")

   ##### INITIALIZATION #####
   def init_ui(self):
      # Set Display
      self.setFixedSize(self._width, self._height)
      self.setWindowTitle(self.title)
      self.main.setFixedSize(self._width - 15, self._height - 15)
      # Create Buttons
      self.init_buttons()
      # SEt Layout
      self.set_layout()
      self.main.show()
      self.show()

   def init_buttons(self):
      
      ##### LABELS #####
      l1 = QLabel('Select Model:')
      l2 = QLabel('Select Audio File:')
      l3 = QLabel('Make a prediction:')
      fname = QLabel('Current File: \"None\"')
      fname.setAlignment(Qt.AlignTop)
      pad = QLabel('')


      ##### BUTTONS #####
      b1 = QPushButton('Load', self.main)
      b1.clicked.connect(self.load_audio)
      b1.setToolTip('Load new Audio File')

      b2 = QPushButton('Playback', self.main)
      b2.clicked.connect(self.playback)
      b2.setToolTip('Playback Current Audio File')

      b3 = QPushButton('Classify', self.main)
      b3.clicked.connect(self.predict)
      b3.setToolTip('Classify Current Audio File')

      b4 = QPushButton('Inspect Features', self.main)
      b4.clicked.connect(self.show_audio_details)
      b4.setToolTip('Get Details on Current Audio Clip')

      b5 = QPushButton('Model Evaluation', self.main)
      b5.clicked.connect(self.show_model_details)
      b5.setToolTip('Display metrics from Model Evaluation')

      # Populate Combo Box with Model Names
      cb1 = QComboBox(self.main)
      for model in self.backend.get_models():
         cb1.addItem(model)
      cb1.currentIndexChanged.connect(self.set_model)
      cb1.setCurrentIndex(0)
      
      self.buttons = {
         'HEAD' : {
            'model_label' : l1,
            'model_list' : cb1,
            'model_details' : b5,

         },
         'DISP' : {
            'audio_label' : l2,
            'audio_load' : b1,
            'audio_play' : b2,
            'audio_details' : b4,
         },

         'TAIL' : {
            'ML_label'   : l3,
            'ML_predict' : b3,
         },

         'UTIL' : {
            'pad_blank' : pad,
            'filename' : fname,
         }
      }
      return 0

   def set_layout(self):
      layout = QVBoxLayout(self.main)
      # Sub Layouts here
      layout_header = QHBoxLayout()
      layout_disp = QHBoxLayout()
      layout_tail = QHBoxLayout()

      # max_widgets = 4
      # for i in range(len(self.buttons['HEAD']), max_widgets):
      #    layout_header.addWidget(self.buttons['UTIL']['pad_blank'])
         
      for button in self.buttons['HEAD'].values():
         layout_header.addWidget(button)
      layout.addLayout(layout_header)

      for button in self.buttons['DISP'].values():
         layout_disp.addWidget(button)
      layout.addLayout(layout_disp)

      layout.addWidget(self.buttons['UTIL']['filename'])

      for button in self.buttons['TAIL'].values():
         layout_tail.addWidget(button)
      layout.addLayout(layout_tail)

      self.main.setLayout(layout)
      return 0

   def update_fname(self):
      path = self.backend.get_path()
      fname = os.path.basename(path)
      display = self.buttons['UTIL']['filename']
      display.setText("Current File: \"" + fname + "\"")
      pass
   
   def dialog(self, msg):
      if not isinstance(msg, str):
         return -1

      mbox = QMessageBox()
      # mbox.setIcon(QMessageBox.information)
      mbox.setWindowTitle('Information:')
      mbox.setText(msg)

      # mbox.setDetailedText(msg)
      mbox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
      mbox.exec_()
      return 0

   def disp(self, msg):
      self.statusBar().showMessage(msg)
      return 0