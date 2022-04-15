import sys, os
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from Model import PixInfo
from View import ResultsDisplay
from Globals import Mode

# main window for User Interaction
class mainUI(QMainWindow):

   # Retrieve and store info from PixInfo passed as param, Build UI
   def __init__(self, pixInfo):
      super().__init__()

      # Main Widet
      self.main = QWidget(self)
      self.setCentralWidget(self.main)

      self.pixInfo = pixInfo
      self.mode = Mode.COLOR

      os.chdir(self.pixInfo.get_imDir())

      self.title = 'CSS 584 - CBIR with RF'
      self.left = 100
      self.top = 100
      self._width = 480
      self._height = 360

      # GUI References
      self.listWidget = QListWidget(self.main)
      self.thumbnail = QLabel(self.main)

      self.initUI()

      # Initialize Side Window (empty)
      self.disp = ResultsDisplay(self)

   # Initialize Userinterface, Display window
   def initUI(self):
      
      # Set Attributes on Application Window
      self.setFixedSize(self._width, self._height)
      self.setWindowTitle(self.title)
      self.main.setFixedSize(self._width - 15, self._height - 15)

      # Initialize buttons
      # Define Window and Layout parameters
      layout = QVBoxLayout()
      lay_head = QHBoxLayout()

      # Inspect Picture
      b1 = QPushButton('View Details', self.main)
      b1.clicked.connect(self.on_details)
      b1.setToolTip('Open selected image with default image viewer')
      lay_head.addWidget(b1)

      # Search button
      b2 = QPushButton('Search', self.main)
      b2.clicked.connect(self.on_search)
      b2.setToolTip('Search directory for related images by \'Mode\'')
      lay_head.addWidget(b2)

      # Search Mode Combo Box [Color / Intensity]
      sub2_vbox = QVBoxLayout()
      label_rb = QLabel('Search Mode:')
      label_rb.setAlignment(Qt.AlignCenter)
      sub2_vbox.addWidget(label_rb)
      sub2_vbox.setSpacing(0)
      
      # Populate Combo Box with Mode Names
      cb1 = QComboBox(self.main)
      for mode in Mode:
         cb1.addItem(mode.name)
      cb1.setCurrentIndex(0)

      # Combo Box Actions
      cb1.currentIndexChanged.connect(self.on_combo)
      cb1.setCurrentIndex(0)
      sub2_vbox.addWidget(cb1)

      # add Horizontal box to vertical layout
      lay_head.addLayout(sub2_vbox)
      layout.addLayout(lay_head)

      # Populate middle of window with Listbox and display
      lay_mid = QHBoxLayout()
      # List-Box
      self.listWidget.setGeometry(0, 0, 85, 200)
      self.listWidget.setFixedWidth(85)

      imList = self.pixInfo.get_imageList()
      # Populate with filenames

      # String to Int to Sort to String
      names = []
      for i in range(len(self.pixInfo.get_imageList())):
         name = os.path.basename(imList[i].filename).split('.')[0]
         # try:
         #    names.append(int(name))
         # except:
         names.append(name)

      names.sort()
      strings = [(str(name) + '.jpg') for name in names]
      self.listWidget.addItems(strings)
      
      # initialize with first item selected and displayed
      self.listWidget.itemClicked.connect(self.on_listChange)
      self.listWidget.setCurrentRow(0)
      self.on_listChange()
      
      # Add widgets to the layout, set layout and show
      lay_mid.addWidget(self.listWidget)
      lay_mid.addWidget(self.thumbnail)
      layout.addLayout(lay_mid)

      self.main.setLayout(layout)
      self.statusBar().showMessage('Initialized')
      self.main.show() 
      self.show()

   # Calculate Manhattan distance between image selected and all other images
   def calc_distance(self):
      imList = self.pixInfo.get_imageList()
      distances = np.zeros(len(imList))
      codes = self.pixInfo.get_codes(self.mode)

      queryIdx = self.listWidget.currentRow()

      # get weights if using Color + Intensity, otherwise neglect
      if self.get_mode() == Mode.COLOR_INTENSITY:
         weights = self.pixInfo.get_weights()
      else:
         weights = np.ones_like(codes[0])

      # Traverse all images
      for compareIdx in range(0,len(imList)):
         dist = 0

         # Traverse all codes
         for i in range(len(codes[queryIdx])):
            dist += weights[i] * abs(codes[queryIdx][i] - codes[compareIdx][i])

         # Add distance to distances array   
         distances[compareIdx] = dist

      # Sort and return array indicating order
      return self.ordered_indices(distances)

   # get index of original array for sorted order
   def ordered_indices(self, original):

      result = original.copy()
      result.sort()

      for i in range(len(original)):
         #search in original, get index
         idx = np.where(original == result[i])
         idx = idx[0][0]

         # reset to negative one to avoid double counting
         original[idx] = -1
         result[i] = idx
      
      return result[1:].astype('int32')

   def refine_search(self):
      self.statusBar().showMessage('Refining Search...')
      self.pixInfo.set_weights(True)
      dist = self.calc_distance()
      self.disp.initWindow(dist)

   # getters
   def get_imageList(self):
      return self.pixInfo.get_imageList()

   def get_photoList(self):
      return self.pixInfo.get_photoList()

   def get_activePath(self):
      return self.activePath

   def get_mode(self):
      return self.mode

   def get_mode_str(self):
      if self.mode == Mode.COLOR:
         modeStr = 'Color'
      elif self.mode == Mode.INTENSITY:
         modeStr = 'Intensity'
      elif self.mode == Mode.COLOR_INTENSITY:
         modeStr = 'Color + Intensity'
      else:
         modeStr = 'Error'
      return modeStr

   def get_checked(self):
      return self.pixInfo.get_checked()

   def set_checked(self, relevant):
      return self.pixInfo.set_checked(relevant)

   def get_size(self):
      return self._width, self._height 

   #setters
   # update file dispalyed for child window
   def set_active(self):
      self.active = self.listWidget.currentItem().text()
      self.activePath = os.path.join(self.pixInfo.get_imDir(), self.active)

   # Event callbacks for user interaction
   # calculate distance, open display, update status bar
   def on_search(self):
      # clear any existing weights if needed
      self.pixInfo.set_weights()
      self.pixInfo.clear_checked()

      # set queuried image to checked
      relevant = np.ones_like(1, shape = len(self.pixInfo.get_checked()))
      relevant *= -1
      relevant[self.listWidget.currentRow()] = 1
      self.pixInfo.set_checked(relevant)

      dist = self.calc_distance()
      self.disp.initWindow(dist)
      self.statusBar().showMessage('Search: ' + self.active + ' | Mode: ' + self.get_mode_str())
   
   # update active mode
   def on_combo(self):
      caller = self.sender()
      self.mode = Mode[caller.currentText()]
      self.statusBar().showMessage('Mode updated: ' + self.get_mode_str())

   # Open the picture with the default operating system image
    # viewer.
   def on_details(self):
      self.statusBar().showMessage('Display details: ' + self.active)
      os.startfile(self.activePath)
   
   # update thumbnail image
   def on_listChange(self):
      self.set_active()
      thumb = QPixmap(self.activePath)
      thumb.scaled(thumb.width() // 4, 
                   thumb.height() // 4,
                   Qt.KeepAspectRatio, 
                   Qt.SmoothTransformation)
      self.thumbnail.setPixmap(thumb)
      self.statusBar().showMessage('Displaying: ' + self.active)
   
def main():
   app = QApplication(sys.argv)
   pixInfo = PixInfo(app)

   mainUI(pixInfo)
   sys.exit(app.exec_())

if __name__ == '__main__':
   main()
