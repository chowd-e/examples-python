import os
from tokenize import Ignore
import numpy as np
from PyQt5.QtWidgets import QLabel, QPushButton, QComboBox, QHBoxLayout 
from PyQt5.QtWidgets import QVBoxLayout, QMainWindow, QWidget, QMessageBox
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QFileDialog as qfd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen

class Viewer(QMainWindow):
   # display search bar for loading file name
   
   def __init__(self, controller, model_parent) -> None:
      super().__init__()
      # Main Widet
      self.main_ = QWidget(self)
      self.setCentralWidget(self.main_)
      
      self.model_ = model_parent
      self.controller_ = controller
      self.painter_ = QPainter()
      self.build()
      
   def send(self, cmd, args=[]):
      return self.controller_.recv(cmd, args)

   def build(self):
      self.title_ = 'Variable Message Sign Detection'
      self.width_ = 600
      self.height_ = 500

      # Button References
      self.buildUI()

   ##### BUTTON RESPONSES #####
   def predict(self):
      resp = self.send('predict')
      # Get key from value in dict
      
      if not isinstance(resp, type(np.zeros(1))):
         self.dialog("Error with prediction, ensure file is loaded")
         return 0
      elif np.all(resp==0):
         # TODO (Howard, Chase) Update Color of contours based on index
         val = "is Not Variable Message Sign"
      else:
         # TODO (Howard, Chase) Update Color of contours based on index in resp
         val = "is Variable Message Sign"

      self.dialog("Prediction: " + val)
      return resp
      
   def drawContour(self):
      self.disp("Drawing Contours...")
      resp = self.send('draw')
      # cast response here

      if resp == None:
         self.dialog("Please load an image first")
      else:
         # TODO (Howard, Chase) Display image, draw contours
         self.disp("Received Contours")
         # contours = resp
         # # Get image
         # im_label = self.buttons_['IMAGE']['display']

         # if im_label.pixmap == None:
         #    self.loadImage()

         # preds = self.predict()

         # # Open Painter object
         # self.painter_ = QPainter(im_label.pixmap)

         # pen_red = QPen(Qt.red)
         # pen_red.setWidth(3)
         # pen_green = QPen(Qt.green)
         # pen_green.setWidth(3)

         # i = 0
         # for prediction in preds:
            
         #    pen = pen_red
         #    if prediction:
         #       pen = pen_green
            
         #    painter.setPen(pen)
         #    painter.drawConvexPolygon(contours[i])
         #    i += 1

   # TODO (Howard, Chase) Display an image on the GUI
   def loadImage(self):
      path = qfd.getOpenFileName(caption="Select image file",
                                 directory="images")[0]
      
      if path == '':
         self.disp("Operation canceled, no Image selected")
         return 0
      
      self.disp('Loading File...')
      resp = self.send('load', path)
      f = os.path.basename(path)

      if resp == 0:
         self.disp("New file loaded: " + str(f))
         self.updateFilename()
         self.displayImage()
      else:
         self.disp("Error loading Image")
      return 0 

   def showDetails(self):
      resp = self.send('details')
      if resp is not None:
         self.disp('Features retrieved')
      else:
         self.dialog('Please load an Image first')
      # resp will be dataframe fo features, how to display
      return resp

   def evaluateModel(self):
      self.disp("Select Source Data")
      source_data = qfd.getOpenFileName(caption="Select Source Data",
                                        directory="data")[0]

      if source_data == '':
         self.disp("Canceled, No data selected to evaluate model perfomance")
         return 0

      return self.send('evaluate', source_data)

   def setModel(self):
      caller = self.sender()
      # Need to get new model with this text
      resp = self.send('load_model', caller.currentText())
      if resp is not None:
         self.disp("Updated Model: " + caller.currentText())
      else:
         self.dialog("Error loading Model")

   ##### INITIALIZATION #####
   def buildUI(self):
      # Set Display
      self.setFixedSize(self.width_, self.height_)
      self.setWindowTitle(self.title_)
      self.main_.setFixedSize(self.width_ - 15, self.height_ - 15)
      # Create Buttons
      self.initializeButtons()
      # SEt Layout
      self.setLayout()
      self.main_.show()
      self.show()

   def initializeButtons(self):
      
      ##### LABELS #####
      l1 = QLabel('Select Model:')
      l2 = QLabel('Select Image File:')
      l3 = QLabel('Make a prediction:')
      l4 = QLabel()
      l4.setMaximumSize(int(self.width_ * 0.95), int(self.height_ * 0.7))

      fname = QLabel('Current File: \"None\"')
      fname.setAlignment(Qt.AlignTop)
      pad = QLabel('')


      ##### BUTTONS #####
      b1 = QPushButton('Load', self.main_)
      b1.clicked.connect(self.loadImage)
      b1.setToolTip('Load new Image')

      b2 = QPushButton('Draw Contours', self.main_)
      b2.clicked.connect(self.drawContour)
      b2.setToolTip('Overlay contours on image')

      b3 = QPushButton('Classify', self.main_)
      b3.clicked.connect(self.predict)
      b3.setToolTip('Classify Current Image')

      b4 = QPushButton('Inspect Features', self.main_)
      b4.clicked.connect(self.showDetails)
      b4.setToolTip('Get Details on Current Image')

      b5 = QPushButton('Model Evaluation', self.main_)
      b5.clicked.connect(self.evaluateModel)
      b5.setToolTip('Display metrics from Model Evaluation')

      # Populate Combo Box with Model Names
      cb1 = QComboBox(self.main_)
      for model in self.model_.getModels():
         cb1.addItem(model)
      cb1.currentIndexChanged.connect(self.setModel)
      cb1.setCurrentIndex(0)
      
      self.buttons_ = {
         'HEAD' : {
            'model_label' : l1,
            'model_list' : cb1,
            'model_details' : b5,

         },
         'DISP' : {
            'image_label' : l2,
            'image_load' : b1,
            'image_draw' : b2,
            'image_details' : b4,
         },
         'IMAGE' : {
            'display' : l4,
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

   def setLayout(self):
      layout = QVBoxLayout(self.main_)
      # Sub Layouts here
      layout_header = QHBoxLayout()
      layout_disp = QHBoxLayout()
      layout_tail = QHBoxLayout()

      for button in self.buttons_['HEAD'].values():
         layout_header.addWidget(button)
      layout.addLayout(layout_header)

      for button in self.buttons_['DISP'].values():
         layout_disp.addWidget(button)
      layout.addLayout(layout_disp)

      layout.addWidget(self.buttons_['IMAGE']['display'])
      layout.addWidget(self.buttons_['UTIL']['filename'])

      for button in self.buttons_['TAIL'].values():
         layout_tail.addWidget(button)
      layout.addLayout(layout_tail)

      self.main_.setLayout(layout)
      return 0

   def updateFilename(self):
      path = self.model_.getPath()
      fname = os.path.basename(path)
      display = self.buttons_['UTIL']['filename']
      display.setText("Current File: \"" + fname + "\"")
      return 0 
   
   def dialog(self, msg):
      if not isinstance(msg, str):
         return -1

      mbox = QMessageBox()
      mbox.setWindowTitle('Information:')
      mbox.setText(msg)

      mbox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
      mbox.exec_()
      return 0

   def displayImage(self):
      im_path = self.model_.getPath()
      if(im_path == None):
         print("Please load an Image first")
         return 0

      im_label = self.buttons_['IMAGE']['display']
      im_label.setPixmap(QPixmap(im_path))
      
   def disp(self, msg):
      self.statusBar().showMessage(msg)
      return 0