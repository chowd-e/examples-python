import sys, os, enum
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from Model import PixInfo
from View import ResultsDisplay

class Mode(enum.Enum):
   COLOR = 0
   INTENSITY = 1

class mainUI(QMainWindow):

   def __init__(self, pixInfo):
      super().__init__()

      # Main Widet
      self.main = QWidget(self)
      self.setCentralWidget(self.main)

      self.pixInfo   = pixInfo
      self.colorCode = pixInfo.get_colorCode()
      self.intenCode = pixInfo.get_intenCode()

      # Full-sized images.
      self.imageList = pixInfo.get_imageList()

      # Thumbnail sized images.
      self.photoList = pixInfo.get_photoList()
      
      # Image size for formatting.
      self.xmax = pixInfo.get_xmax()
      self.ymax = pixInfo.get_ymax()
      self.mode = Mode.COLOR
      
      # set path based on exe or not
      if getattr(sys, 'frozen', False):
         app_path = os.path.dirname(sys.executable)
      else:
         app_path = os.path.dirname(os.path.abspath(__file__))

      self.imDir = os.path.join(app_path, 'images')

      self.title = 'CSS 584 - HW1 - Chase Howard'
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

      os.chdir(self.imDir)

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

      # Search Mode Radio Buttons [Color / Intensity]
      sub2_vbox = QVBoxLayout()
      label_rb = QLabel('Search Mode:')
      sub2_vbox.addWidget(label_rb)
      sub2_vbox.setSpacing(0)
      
      rb1 = QRadioButton('Color', self.main)
      rb1.toggled.connect(self.on_radio)
      rb1.setChecked(True)
      sub2_vbox.addWidget(rb1)
      
      rb2 = QRadioButton('Intensity', self.main)
      rb2.toggled.connect(self.on_radio)
      sub2_vbox.addWidget(rb2)

      # add Horizontal box to vertical layout
      lay_head.addLayout(sub2_vbox)
      layout.addLayout(lay_head)

      # Populate middle of window with Listbox and display
      lay_mid = QHBoxLayout()
      # List-Box
      self.listWidget.setGeometry(0, 0, 85, 200)
      self.listWidget.setFixedWidth(85)

      # Populate with filenames
      for i in range(len(self.imageList)):
         name = os.path.basename(self.imageList[i].filename)
         self.listWidget.addItem(name)
      
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

   # update file dispalyed for child window
   def update_active(self):
      self.active = self.listWidget.currentItem().text()
      self.activePath = os.path.join(self.imDir, self.active)
      
   # Open the picture with the default operating system image
    # viewer.
   def on_details(self):
      self.statusBar().showMessage('Display details: ' + self.active)
      os.startfile(self.activePath)
   
   # update thumbnail image
   def on_listChange(self):
      self.update_active()
      thumb = QPixmap(self.activePath)
      thumb.scaled(thumb.width() // 4, 
                   thumb.height() // 4,
                   Qt.KeepAspectRatio, 
                   Qt.SmoothTransformation)
      self.thumbnail.setPixmap(thumb)
      self.statusBar().showMessage('Displaying: ' + self.active)
   
   # Calculate Manhattan distance between image selected and all other images
   def calc_distance(self):
      distances = [0] * (len(self.imageList))
      
      if self.mode == Mode.COLOR:
         bins = self.colorCode
      else:
         bins = self.intenCode

      # Traverse all images
      queryIdx = self.listWidget.currentRow()
      for compareIdx in range(0,len(self.imageList)):

         x, y = self.photoList[queryIdx].size
         querySz = x * y

         x, y = self.photoList[compareIdx].size
         compareSz = x * y

         dist = 0
         # Traverse all bins
         for i in range(len(bins[queryIdx])):
            dist += abs((bins[queryIdx][i] / querySz) - (bins[compareIdx][i] / compareSz))
            
         distances[compareIdx] = dist
         # sort distances in ascending order
         sortDist = distances.copy()
         sortDist.sort()

      # find min, in place
      # replace distances with order of dist
      for i in range(len(sortDist)):
         #search in Distances, get index
         minIdx = distances.index(sortDist[i])
         # reset to negative one to avoid double counting
         distances[minIdx] = -1
         sortDist[i] = minIdx
      return sortDist[1:]

   # getters
   def get_imageList(self):
      return self.imageList

   def get_photoList(self):
      return self.photoList

   def get_activePath(self):
      return self.activePath

   def get_mode(self):
      return self.mode

   def get_size(self):
      return self._width, self._height 

   # Event callbacks for user interaction
   # calculate distance, open display, update status bar
   def on_search(self):
      dist = self.calc_distance()
      self.disp.initWindow(dist)
      modeStr = str(self.mode).split('.')[-1]
      self.statusBar().showMessage('Search: ' + self.active + ' | Mode: ' + modeStr)
   
   # update active mode
   def on_radio(self):
      caller = self.sender()
      if(caller.isChecked()):
         if caller.text() == 'Color':
            self.mode = Mode.COLOR
         else:
            self.mode = Mode.INTENSITY
      modeStr = str(self.mode).split('.')[-1]
      self.statusBar().showMessage('Mode updated: ' + modeStr)
   
def main():
   app = QApplication(sys.argv)
   pixInfo = PixInfo(app)

   mainUI(pixInfo)
   sys.exit(app.exec_())

if __name__ == '__main__':
   main()
