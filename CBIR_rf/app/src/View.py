import os
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap

from Globals import Mode

class ResultsDisplay(QWidget):
   # Initialize to empty
   disp = None

   def __init__(self, parent):
      super().__init__()
      self.controller = parent

      # Bolded font
      self.bolded = QFont()
      self.bolded.setBold(True)
      self.bolded.setPointSize(12)

      w, h = self.controller.get_size()
      self._width = w * 1.2
      self._height = h * 1.5

      
      # Set Grid params
      self.gridX = 5
      self.gridY = 4
      self.gridIm = []
      self.gridRfCheck = []
      self.gridLabels = []

      # Initialize Push Buttons
      self.prev = QPushButton('Prev', self.disp)
      self.next = QPushButton('Next', self.disp)
      self.refine = QPushButton('Refine Search', self.disp)

      # main Layout
      self.lay = QVBoxLayout()
      # sub layouts
      self.head = QHBoxLayout()
      self.grid = QGridLayout()
      self.tail = QHBoxLayout()

      # Queried Image
      self.queried = QLabel(self.disp)

   def initWindow(self, dist):
      # If the window exists, don't recreate, just open. [Initialized as null]
      if not self.disp:
         self.disp = QWidget()
         self.distances = dist

         # don't put this in init_tail, need page num and limit for init_grid
         # number of pages in grid for navigating
         self.pageNum = 0
         iq = len(self.distances) // (self.gridX * self.gridY)
         # Add a page if there is a remainder
         if iq % 1 != 0:
            self.pageLim = iq + 1
         else:
            self.pageLim = iq

         self.init_head()
         self.init_grid()
         self.init_tail()

         self.disp.setLayout(self.lay)
         self.disp.setGeometry(0, 0, int(self._width), int(self._height))
      else:
         # Already open, Reset and reload page
         self.distances = dist
         self.pageNum = 0
         self.set_head()
         self.set_grid()
         self.set_tail()
      
      self.disp.show()
      self.disp.activateWindow()

   # BUIld header on GUI
   def init_head(self):
      # buffer values to preserve proper spacing
      lay_buff = QVBoxLayout()
      buff = QLabel(self.disp)

      self.refine.clicked.connect(self.on_refine)

      self.headLabel = QLabel(self.disp)
      self.headLabel.setFont(self.bolded)
      self.headLabel.setAlignment(Qt.AlignRight)
      self.queried.setAlignment(Qt.AlignLeft)

      # Label for Searched image displayed at top with Mode
      lay_buff.addWidget(self.headLabel)
      lay_buff.addWidget(self.refine)
      lay_buff.addWidget(buff)
      lay_buff.addWidget(buff)

      # self.head.addWidget(buff)
      self.head.addWidget(buff)
      self.head.addWidget(buff)
      self.head.addLayout(lay_buff)

      self.head.addWidget(self.queried)
      self.lay.addLayout(self.head)
      self.set_head()

   # initialize grid with Qlable placeholders for each image
   def init_grid(self):
      self.grid = QGridLayout()
      self.grid.setSpacing(4)
      for i in range(self.gridY):
         for j in range(self.gridX):
               box = QVBoxLayout()
               tail = QHBoxLayout()

               # Image Display
               im_label = QLabel(self.disp)
               im_label.setStyleSheet('border: 1px solid black;')
               im_label.setAlignment(Qt.AlignCenter)
               self.gridIm.append(im_label)

               # Label display image name
               im_name = QLabel(self.disp)
               im_name.setAlignment(Qt.AlignRight)
               im_name.setAlignment(Qt.AlignVCenter)
               self.gridLabels.append(im_name)

               # Relevant Checkbox 
               rf_check = QCheckBox(self.disp)
               rf_check.setText('Relevant')
               self.gridRfCheck.append(rf_check)

               tail.addWidget(rf_check)
               tail.addWidget(im_name)
               box.addWidget(im_label)
               box.addLayout(tail)
               self.grid.addLayout(box, i, j)            

      self.lay.addLayout(self.grid)
      self.set_grid()

   # Initialize bottom section of GUI containing Next / Prev buttons 
   # with page display
   def init_tail(self):
      self.prev.clicked.connect(self.on_prev)
      self.next.clicked.connect(self.on_next)
      self.pageLabel = QLabel(self.disp)
      self.pageLabel.setAlignment(Qt.AlignCenter)
      buffLeft = QLabel(self.disp)
      buffRight = QLabel(self.disp)

      self.tail.addWidget(self.prev)
      self.tail.addWidget(buffLeft)
      self.tail.addWidget(self.pageLabel)
      self.tail.addWidget(buffRight)
      self.tail.addWidget(self.next)
      self.lay.addLayout(self.tail)
      self.set_tail()

   
   # setters:
   # Refresh header due to user change on controller
   def set_head(self):
      # Update Queried Image
      filePath = self.controller.get_activePath()
      im = QPixmap(filePath)
      im = im.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
      self.queried.setPixmap(im)

      # show or hide Refine Button based on operating Mode
      if(self.controller.get_mode() != Mode.COLOR_INTENSITY):
         self.refine.hide()
      else:
         self.refine.show()

      # Update text label
      tail = os.path.basename(filePath)

      modeStr = self.controller.get_mode_str()
      contents = "Queried Image: " + tail + "\nMethod: " + modeStr
      self.headLabel.setText(contents)
      self.disp.setWindowTitle('Search results: ' + tail)

   # 4x5 grid of thumbnail images sorted by dist (left -> right, top -> bot)
   def set_grid(self):
      
      # Ranges from 0 - Total # Images
      idx = self.pageNum * self.gridX * self.gridY
      photos = self.controller.get_photoList()
      isChecked = self.controller.get_checked()

      for i in range(self.gridY * self.gridX):
         
         img = self.gridIm[i]
         name = self.gridLabels[i]
         rf = self.gridRfCheck[i]

         # Re-populate image
         if idx < len(self.distances):
            im_idx = self.distances[idx]
            pil = photos[im_idx]
            idx += 1
            im = QPixmap(pil.filename)
            im = im.scaled(150, 
                           150, 
                           Qt.KeepAspectRatio, 
                           Qt.SmoothTransformation)
            img.setPixmap(im)

            # Show or hide relevant checkbox based on operating mode
            rf.setChecked(isChecked[im_idx])
            if(self.controller.get_mode() != Mode.COLOR_INTENSITY):
               rf.hide()
            else:
               rf.show()

            tail = os.path.basename(pil.filename)
            name.setText(tail)
            img.setToolTip(tail)
         else:
            # remove image from grid display
            img.clear()
            name.hide()
            rf.hide()

   # Update bottom section of display due to UI action
   def set_tail(self):
      tail = '/' + str(self.pageLim) + ')'

      prev = str(self.pageNum - 1)
      next = str(self.pageNum + 1)

      self.pageLabel.setText('Current Page: ' + str(self.pageNum))
      if self.pageNum == 0:
         prev = '~'
      if self.pageNum == self.pageLim:
         next = '~'
      self.prev.setText('Prev ('+ prev + tail)
      self.next.setText('Next ('+ next + tail)

   # Actions on button press: 
   def on_prev(self):
      if self.pageNum != 0:
         self.set_checked()
         self.pageNum -= 1
         self.set_grid()
         self.set_tail()

   def set_checked(self):
      # Set to -1 for no change, 0 for no check, 1 for check
      update = np.ones_like(self.distances, shape = len(self.distances) + 1)
      update *= -1

      pageLen = self.gridX * self.gridY
      start = self.pageNum * pageLen
      stop = start + pageLen

      gridTraverse = 0
      for idx in range(start, stop):
         if idx >= len(self.distances):
            continue
         im_idx = self.distances[idx]
         if self.gridRfCheck[gridTraverse].isChecked():
            update[im_idx] = 1
         else:
            update[im_idx] = 0
         gridTraverse += 1
      
      self.controller.set_checked(update)
      return 0

   # callback functions
   def on_next(self):
      if self.pageNum < self.pageLim:
         self.set_checked()
         self.pageNum += 1
         self.set_grid()
         self.set_tail()

   def on_refine(self): 
      self.set_checked()
      self.pageNum = 0
      self.controller.refine_search()