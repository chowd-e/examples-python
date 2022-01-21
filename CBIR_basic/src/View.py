import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap

class ResultsDisplay(QWidget):
   # Initialize to empty
   disp = None

   def __init__(self, parent):
      super().__init__()
      self.viewer = parent
      self.photos = self.viewer.get_photoList()

      # Bolded font
      self.bolded = QFont()
      self.bolded.setBold(True)
      self.bolded.setPointSize(12)

      w, h = self.viewer.get_size()
      self._width = w * 1.2
      self._height = h * 1.5

      # Set Grid params
      self.gridX = 5
      self.gridY = 4
      self.gridLabels = []

      # Initialize Push Buttons
      self.prev = QPushButton('Prev', self.disp)
      self.next = QPushButton('Next', self.disp)

      # main Layout
      self.lay = QVBoxLayout()
      # sub layouts
      self.head = QHBoxLayout()
      self.grid = QGridLayout()
      self.tail = QHBoxLayout()

      # Queried Image
      self.queried = QLabel(self.disp)

   def initWindow(self, dist):
      # If the window exists, don't recreate, just open
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
         self.update_head()
         self.update_grid()
         self.update_tail()
      
      self.disp.show()
      self.disp.activateWindow()

   def init_head(self):
      lay_buff = QVBoxLayout()
      buff = QLabel(self.disp)
      # botBuff = QLabel(self.disp)

      self.headLabel = QLabel(self.disp)
      self.headLabel.setFont(self.bolded)
      self.headLabel.setAlignment(Qt.AlignRight)
      self.queried.setAlignment(Qt.AlignLeft)
      # Label for Searched image displayed at top with Mode
      lay_buff.addWidget(buff)
      lay_buff.addWidget(self.headLabel)
      lay_buff.addWidget(buff)
      self.head.addWidget(buff)
      self.head.addLayout(lay_buff)

      self.head.addWidget(self.queried)
      self.lay.addLayout(self.head)
      self.update_head()

   def update_head(self):
      # Update Queried Image
      filePath = self.viewer.get_activePath()
      im = QPixmap(filePath)
      im = im.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
      self.queried.setPixmap(im)

      # Update text label
      tail = os.path.basename(filePath)
      modeStr = str(self.viewer.get_mode()).split('.')[-1]
      contents = "Queried Image: " + tail + "\nSearch type: " + modeStr
      self.headLabel.setText(contents)
      self.disp.setWindowTitle('Search results: ' + tail)

   # initialize grid with Qlable placeholders for each image
   def init_grid(self):
      self.grid = QGridLayout()
      self.grid.setSpacing(0)
      for i in range(self.gridY):
         for j in range(self.gridX):
               im_label = QLabel(self.disp)
               im_label.setStyleSheet('border: 1px solid black;')
               im_label.setAlignment(Qt.AlignCenter)
               self.gridLabels.append(im_label)
               self.grid.addWidget(im_label, i, j)

      self.lay.addLayout(self.grid)
      self.update_grid()

   # 4x5 grid of thumbnail images sorted by dist (left -> right, top -> bot)
   def update_grid(self):
      
      # Ranges from 0 - Total # Images
      idx = self.pageNum * self.gridX * self.gridY
      
      for i in range(self.gridY * self.gridX):
         
         im_label = self.gridLabels[i]
         # Re-populate image
         if idx < len(self.distances):
            pil = self.photos[self.distances[idx]]
            idx += 1
            im = QPixmap(pil.filename)
            im = im.scaled(150, 
                           150, 
                           Qt.KeepAspectRatio, 
                           Qt.SmoothTransformation)
            im_label.setPixmap(im)

            tail = os.path.basename(pil.filename)
            im_label.setToolTip(tail)
         else:
            # remove image from grid display
            im_label.clear()

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
      self.update_tail()

   def update_tail(self):
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

   def on_prev(self):
      if self.pageNum != 0:
         self.pageNum -= 1
         self.update_grid()
         self.update_tail()

   def on_next(self):
      if self.pageNum < self.pageLim:
         self.pageNum += 1
         self.update_grid()
         self.update_tail()