#Main script for running QuantiFly training.
import time
import numpy as np
from PyQt4 import QtGui, QtCore,QtWebKit
import errno
import os
import os.path
import re
import cPickle as pickle
import sys

from scipy.special import _ufuncs_cxx
import sklearn.utils.lgamma
from gnu import return_license
import matplotlib.lines as lines
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import v2_functions as v2





"""QuantiFly Software v2.0

    Copyright (C) 2015  Dominic Waithe

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""


class fileDialog(QtGui.QMainWindow):
    """The dialog for loading images"""
    def __init__(self,parent):
        super(fileDialog, self).__init__()
        self.parent = parent
        self.initUI()
        self.parent.config ={}

        try:
            self.parent.config = pickle.load(open(os.path.expanduser('~')+'/.densitycount/config.p', "rb" ));
            self.parent.filepath = self.parent.config['filepath']
        except:
            self.parent.filepath = os.path.expanduser('~')+'/'
            try:
                os.makedirs(os.path.expanduser('~')+'/.densitycount/')
            except:
                'unable to make directory: ',os.path.expanduser('~')+'/.densitycount/'

        
    def initUI(self):      

        self.textEdit = QtGui.QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.statusBar()

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open', self)

        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)       
        
        self.setGeometry(300, 300, 350, 500)
        self.setWindowTitle('File dialog')
        #self.show()
        
    def showDialog(self):
        par_obj.file_array =[]
        par_obj.feat_arr ={}
        par_obj.RF ={}

        self.parent.selIntButton.setEnabled(False)
        #filepath = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        for path in QtGui.QFileDialog.getOpenFileNames(self, 'Open file',self.parent.filepath,'Images(*.tif *.tiff *.png);;'):
            par_obj.file_array.append(path)

            
            self.parent.config['filepath'] = str(QtCore.QFileInfo(path).absolutePath())+'/'
            pickle.dump(self.parent.config, open(str(os.path.expanduser('~')+'/.densitycount/config.p'), "w" ))
            
        self.parent.image_status_text.showMessage('Status: Loading Images. ')                
        success, updateText = v2.import_data_fn(par_obj, par_obj.file_array)
        #loadWin.CH_cbx1.stateChange()
        
        self.parent.image_status_text.showMessage(updateText)
        if success == True:
            self.parent.updateAfterImport()



        
class Load_win_fn(QtGui.QWidget):
    """The class for loading and processing images"""
    def __init__(self,par_obj,win):
        super(Load_win_fn, self).__init__()
        #Vertical layout
        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)
        hbox0 = QtGui.QHBoxLayout()
        vbox.addLayout(hbox0)

        #Load images button
        self.loadImages_button = QtGui.QPushButton("Load Images", self)
        hbox0.addWidget(self.loadImages_button)
        hbox0.addStretch()

        
        about_btn = QtGui.QPushButton('About')
        about_btn.clicked.connect(self.on_about)
        hbox0.addWidget(about_btn)

        #Load button.
        self.Text_CHopt = QtGui.QLabel()
        vbox.addWidget(self.Text_CHopt)
        self.ex = fileDialog(self)
        self.loadImages_button.clicked.connect(self.ex.showDialog)
        
        
        #SigmaData input field.
        self.feature_scale_input = QtGui.QLineEdit(str(par_obj.feature_scale))
        #SigmaData input Text.
        self.feature_scaleText = QtGui.QLabel()
        
        hbox1 = QtGui.QHBoxLayout()
        vbox.addWidget(self.feature_scaleText)
        vbox.addLayout(hbox1)
        
        
        self.feature_scaleText.resize(40,20)
        self.feature_scaleText.setText('Input sigma for feature calculation default (0.8):')
        self.feature_scaleText.hide()

        hbox1.addWidget(self.feature_scale_input)
        hbox1.addStretch()
        
        self.feature_scale_input.resize(10,10)
        self.feature_scale_input.textChanged[str].connect(self.feature_scale_change)
        self.feature_scale_input.hide()

        
        
        CH_pX = 0
        CH_pY = 50

        #Channel dialog generation.
        
        vbox.addWidget(self.Text_CHopt)
        self.Text_CHopt.setText('Please select which channels you want to include in the training:')
        self.Text_CHopt.resize(500,40)
        self.Text_CHopt.hide()

        
        hbox2 = QtGui.QHBoxLayout()           
        #Object factory for channel selection.
        for i in range(0,10):
            name = 'self.CH_cbx'+str(i+1)+'_txt = QtGui.QLabel()'
            exec(name)
            name = 'self.CH_cbx'+str(i+1)+'_txt.setText(\'CH '+str(i+1)+':\')'
            exec(name)
            name = 'hbox2.addWidget(self.CH_cbx'+str(i+1)+'_txt)'
            exec(name)
            name = 'self.CH_cbx'+str(i+1)+'_txt.hide()'
            exec(name)
            name = 'self.CH_cbx'+str(i+1)+'= checkBoxCH()'
            exec(name)
            name = 'hbox2.addWidget(self.CH_cbx'+str(i+1)+')'
            exec(name)
            name = 'self.CH_cbx'+str(i+1)+'.hide()'
            exec(name)
            name = 'self.CH_cbx'+str(i+1)+'.setChecked(True)'
            exec(name)
            name = 'self.CH_cbx'+str(i+1)+'.type =\'feature_ch\''
            exec(name)
        hbox2.addStretch()
        vbox.addLayout(hbox2)


        self.figure1 = Figure(figsize=(2,2))
        self.canvas1 = FigureCanvas(self.figure1)
        

        #self.figure1.patch.set_facecolor('white')
        self.plt1 = self.figure1.add_subplot(1,1,1)
        self.resize(100,100)
        self.canvas1.hide()
        self.figure1.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])
        
        hbox3 = QtGui.QHBoxLayout()
        vbox.addLayout(hbox3)
        hbox3.addWidget(self.canvas1)
        hbox3.addStretch()

        #Channel dialog generation.
        self.Text_FrmOpt2 = QtGui.QLabel()
        vbox.addWidget(self.Text_FrmOpt2)
        self.Text_FrmOpt2.hide()

        #Image frames dialog.
        Text_FrmOpt1_panel = QtGui.QHBoxLayout()
        self.Text_FrmOpt1 = QtGui.QLabel()
        self.Text_FrmOpt1.setText('Please choose the frames you wish to use for training. Use either \',\' to separate individual frames or a \'-\' to indicate a range:')
        self.Text_FrmOpt1.hide()
        Text_FrmOpt1_panel.addWidget(self.Text_FrmOpt1)
        vbox.addLayout(Text_FrmOpt1_panel)

        #Image frames input.
        linEdit_Frm_panel = QtGui.QHBoxLayout()
        self.linEdit_Frm = QtGui.QLineEdit()
        self.linEdit_Frm.hide()
        linEdit_Frm_panel.addWidget(self.linEdit_Frm)
        linEdit_Frm_panel.addStretch()
        vbox.addLayout(linEdit_Frm_panel)
        

        #Feature calculation to perform:
        self.Text_Radio = QtGui.QLabel()
        vbox.addWidget(self.Text_Radio)
        self.Text_Radio.setText('Feature select which kind of feature detection you would like to use:')
        self.Text_Radio.resize(500,40)
        self.Text_Radio.hide()
        self.radio_group=QtGui.QButtonGroup(self) # Number 
        self.r0 = QtGui.QRadioButton("Basic",self)
        self.r1 = QtGui.QRadioButton("Fine",self)
        
        self.r1.setChecked(True)
        self.radio_group.addButton(self.r0)
        self.radio_group.addButton(self.r1)
        
        vbox.addWidget(self.r0)
        vbox.addWidget(self.r1)
        
        self.r0.hide()
        self.r1.hide()
        vbox.addStretch()
        #Green status text.
        
        #Load images button
        hbox1 = QtGui.QHBoxLayout()
        vbox.addLayout(hbox1)
        self.confirmImages_button = QtGui.QPushButton("Confirm Images")
        hbox1.addWidget(self.confirmImages_button)
        self.confirmImages_button.clicked.connect(self.processImgs)
        self.confirmImages_button.setEnabled(False)
        #Move to training button.
        self.selIntButton = QtGui.QPushButton("Goto Training")
        hbox1.addWidget(self.selIntButton)
        self.selIntButton.clicked.connect(win.loadTrainFn)
        self.selIntButton.setEnabled(False)

        self.image_status_text = QtGui.QStatusBar()
        self.image_status_text.setStyleSheet("QLabel {  color : green }")
        self.image_status_text.showMessage('Status: Highlight training images in folder. ')
        vbox.addWidget(self.image_status_text)
        
        
        hbox1.addStretch()

       
        #File browsing functions
        layout = QtGui.QVBoxLayout() 
    def on_about(self):
        self.about_win = QtGui.QWidget()
        self.about_win.setWindowTitle('About QuantiFly Software v2.0')
        
        license = return_license()
        #with open (sys.path[0]+'/GNU GENERAL PUBLIC LICENSE.txt', "r") as myfile:
        #    data=myfile.read().replace('\n', ' ')
        #    license.append(data)
        
        # And give it a layout
        layout = QtGui.QVBoxLayout()
        
        self.view = QtWebKit.QWebView()
        self.view.setHtml('''
          <html>
            
         
            <body>
              <form>
                <h1 >About</h1>
                
                <p>Software written by Dominic Waithe (c) 2015</p>
                '''+str(license)+'''
                
                
              </form>
            </body>
          </html>
        ''')
        layout.addWidget(self.view)
        self.about_win.setLayout(layout)
        self.about_win.show()
        self.about_win.raise_()   

    def feature_scale_change(self,text):
        """Updates on change of feature scale"""
        par_obj.feature_scale = float(text)

    def updateAfterImport(self):
        """Specific to ui updates"""

        if par_obj.file_ext == 'tif' or par_obj.file_ext == 'tiff':
            if par_obj.tiff_file.maxFrames >1:
                self.linEdit_Frm.setText('1-'+str(par_obj.uploadLimit))
                self.Text_FrmOpt2.setText('There are '+str(par_obj.tiff_file.maxFrames)+' frames in total.')
                self.Text_FrmOpt1.show()
                self.Text_FrmOpt2.show()
                self.linEdit_Frm.show()

        self.confirmImages_button.setEnabled(True)
        
        self.plt1.cla()
        self.plt1.imshow(255-par_obj.ex_img)
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])
        self.canvas1.show()
        self.canvas1.draw()
       
        
        par_obj.ch_active =[];
        if par_obj.numCH> 2:
            self.Text_CHopt.show()
            for i in range(0,par_obj.numCH):
                name = 'self.CH_cbx'+str(i+1)+'.show()'
                exec(name)
                name = 'self.CH_cbx'+str(i+1)+'_txt.show()'
                exec(name)
                par_obj.ch_active.append(i)
        else:
            par_obj.ch_active.append(0)
        
        self.feature_scale_input.show()
        self.feature_scaleText.show()
        self.r0.show()
        self.r1.show()
        self.Text_Radio.show()
    def hyphen_range(self,s):
        """ yield each integer from a complex range string like "1-9,12, 15-20,23"

        >>> list(hyphen_range('1-9,12, 15-20,23'))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 23]

        >>> list(hyphen_range('1-9,12, 15-20,2-3-4'))
        Traceback (most recent call last):
            ...
        ValueError: format error in 2-3-4
        """
        for x in s.split(','):
            elem = x.split('-')
            if len(elem) == 1: # a number
                yield int(elem[0])
            elif len(elem) == 2: # a range inclusive
                start, end = map(int, elem)
                for i in xrange(start, end+1):
                    yield i
            else: # more than one hyphen
                raise ValueError('format error in %s' % x)


    def processImgs(self):
        """Loads images and calculates the features."""
        #Resets everything should this be another patch of images loaded.
        imgs =[]
        gt_im_sing_chgs =[]
        fmStr = self.linEdit_Frm.text()
        par_obj.feat_arr ={}
        par_obj.pred_arr ={}
        par_obj.sum_pred ={}
        par_obj.frames_2_load ={}
        par_obj.left_2_calc =[]
        par_obj.saved_ROI =[]
        par_obj.saved_dots=[]
        par_obj.curr_img = 0
        par_obj.eval_load_im_win_eval = False

        if self.r0.isChecked():
            par_obj.feature_type = 'basic'
        if self.r1.isChecked():
            par_obj.feature_type = 'fine'
   
        #Now we commit our options to our imported files.
        if par_obj.file_ext == 'png':
            for i in range(0,par_obj.file_array.__len__()):
                par_obj.left_2_calc.append(i)
                par_obj.frames_2_load[i] = [0]
            self.image_status_text.showMessage('Status: Loading Images. Loading Image Num: '+str(par_obj.file_array.__len__()))
                
            v2.im_pred_inline_fn(par_obj, self)
        elif par_obj.file_ext =='tiff' or par_obj.file_ext =='tif':
            if par_obj.tiff_file.maxFrames>1:
                for i in range(0,par_obj.file_array.__len__()):
                    par_obj.left_2_calc.append(i)
                    try:
                        np.array(list(self.hyphen_range(fmStr)))-1
                        par_obj.frames_2_load[i] = np.array(list(self.hyphen_range(fmStr)))-1
                    except:
                        self.image_status_text.showMessage('Status: The supplied range of image frames is in the wrong format. Please correct and click confirm images.')
                        return
                self.image_status_text.showMessage('Status: Loading Images.')
                v2.im_pred_inline_fn(par_obj, self)
               
            else:
                for i in range(0,par_obj.file_array.__len__()):
                    par_obj.left_2_calc.append(i)
                    par_obj.frames_2_load[i] = [0]
                v2.im_pred_inline_fn(par_obj, self)
        count = 0
        for b in par_obj.left_2_calc:
            frames =par_obj.frames_2_load[b]
            for i in frames:
                count = count+1
                
        par_obj.test_im_start= 0
        par_obj.test_im_end= count
        
        
        for i in par_obj.left_2_calc:
            im_array = np.zeros((par_obj.height,par_obj.width))
            par_obj.dense_array[i]=im_array
        
        self.image_status_text.showMessage('Status: Images loaded. Click \'Goto Training\'')
        self.selIntButton.setEnabled(True)
        par_obj.imgs = imgs
    def report_progress(self,message):
        self.image_status_text.showMessage('Status: '+message)
        app.processEvents()

class Win_fn(QtGui.QWidget):
    """Class which houses main training functionality"""
    def __init__(self,par_obj):
        super(Win_fn, self).__init__()
        #Sets up the figures for displaying images.
        self.figure1 = Figure(figsize=(8, 8), dpi=100)
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure1.patch.set_facecolor('grey')
        
        
        self.plt1 = self.figure1.add_subplot(1, 1, 1)
        im_RGB = np.zeros((512, 512))
        #Makes sure it spans the whole figure.
        self.figure1.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)
        
        self.plt1.imshow(im_RGB)

        #Removes the tick labels
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])
        
        #Initialises the second figure.
        self.figure2 = Figure(figsize=(8, 8), dpi=100)
        self.canvas2 = FigureCanvas(self.figure2)
        self.figure2.patch.set_facecolor('grey')
        self.plt2 = self.figure2.add_subplot(1, 1, 1)
        
        #Makes sure it spans the whole figure.
        self.figure2.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)
        self.plt2.imshow(im_RGB)
        self.plt2.set_xticklabels([])
        self.plt2.set_yticklabels([])
        
        #The ui for training
        self.count_txt = QtGui.QLabel()
        self.image_num_txt = QtGui.QLabel()
        box = QtGui.QVBoxLayout()
        self.setLayout(box)
        
        #Widget containing the top panel.
        top_panel = QtGui.QHBoxLayout()
        
        #Top left and right widget panels
        top_left_panel = QtGui.QGroupBox('Basic Controls')
        top_right_panel = QtGui.QGroupBox('Advanced Controls')
        
        #Grid layouts for the top and left panels.
        self.top_left_grid = QtGui.QGridLayout()
        self.top_right_grid = QtGui.QGridLayout()
        self.top_left_grid.setSpacing(2)
        self.top_right_grid.setSpacing(2)
        
        #Widgets for the top panel.
        top_panel.addWidget(top_left_panel)
        top_panel.addWidget(top_right_panel)
        top_panel.addStretch()
        
        #Set the layout of the panels to be the grids.
        top_left_panel.setLayout(self.top_left_grid)
        top_right_panel.setLayout(self.top_right_grid)
        
        #Sets up the button which changes to the prev image
        self.prev_im_btn = QtGui.QPushButton('Prev Image')
        self.prev_im_btn.setEnabled(True)
        
        #Sets up the button which changes to the next Image.
        self.next_im_btn = QtGui.QPushButton('Next Image')
        self.next_im_btn.setEnabled(True)
        
        #Sets the current text.
        self.image_num_txt.setText('The Current Image is: ' + str(par_obj.curr_img +1))
        self.count_txt = QtGui.QLabel()
        
        #Sets up the button which saves the ROI.
        self.save_ROI_btn = QtGui.QPushButton('Save ROI')
        self.save_ROI_btn.setEnabled(True)
        
        #Sets up the button which saves the ROI.
        self.save_dots_btn = QtGui.QPushButton('Save Dots')
        self.save_dots_btn.setEnabled(False)
        
        #Button for training model
        self.train_model_btn = QtGui.QPushButton('Train Model')
        self.train_model_btn.setEnabled(False) 
        
        #Selects and reactivates an existing ROI.
        self.sel_ROI_btn = QtGui.QPushButton('Select ROI')
        self.sel_ROI_btn.setEnabled(True)
        self.select_ROI= False
        
        #Allows deletion of dots.
        self.remove_dots_btn = QtGui.QPushButton('Remove Dots')
        self.remove_dots_btn.setEnabled(False)
        self.remove_dots = False
        
        #Populates the grid with the different widgets.
        self.top_left_grid.addWidget(self.prev_im_btn, 0, 0)
        self.top_left_grid.addWidget(self.next_im_btn, 0, 1)
        self.top_left_grid.addWidget(self.image_num_txt, 2, 0, 2, 3)
        self.top_left_grid.addWidget(self.save_ROI_btn, 4, 0)
        self.top_left_grid.addWidget(self.save_dots_btn, 4, 1)
        self.top_left_grid.addWidget(self.train_model_btn, 4, 2)
        self.top_left_grid.addWidget(self.sel_ROI_btn, 5, 0)
        self.top_left_grid.addWidget(self.remove_dots_btn, 5, 1)

        #SigmaData input Label.
        self.sigma_data_text = QtGui.QLabel(self)
        self.sigma_data_text.setText('Input Sigma for Kernel Size:')
        self.top_right_grid.addWidget(self.sigma_data_text, 0, 0)

        #SigmaData input field.
        self.sigma_data_input = QtGui.QLineEdit(str(par_obj.sigma_data))
        self.sigma_data_input.onChanged = self.sigmaOnChange
        self.sigma_data_input.setFixedWidth(40)
        self.sigma_data_input.textChanged[str].connect(self.sigmaOnChange)
        self.top_right_grid.addWidget(self.sigma_data_input, 0, 1)

        #Feature scale input Label.
        #self.sigma_data_text = QtGui.QLabel()
        #self.sigma_data_text.setText('Scale of Feature Descriptor:')
        #self.top_right_grid.addWidget(self.sigma_data_text, 1, 0)
        
        #Feature scale input field
        #self.feature_scale_input = QtGui.QLineEdit(str(par_obj.feature_scale))
        #self.feature_scale_input.onChanged = self.feature_scale_change
        #self.feature_scale_input.resize(40, 20)
        #self.feature_scale_input.textChanged[str].connect(self.feature_scale_change)
        #self.feature_scale_input.setFixedWidth(40)
        #self.top_right_grid.addWidget(self.feature_scale_input, 1, 1)
        
        #Feature scale input btn.
        #self.feat_scale_change_btn = QtGui.QPushButton('Recalculate Features')
        #self.feat_scale_change_btn.setEnabled(True)
        #self.top_right_grid.addWidget(self.feat_scale_change_btn, 1, 2)
        
        #Saves the model
        self.save_model_btn = QtGui.QPushButton('Save Training Model')
        self.save_model_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.save_model_btn, 1, 0)
        
        #Saves the extremel random decision tree model
        self.save_model_name_txt = QtGui.QLineEdit('Insert Model Name')
        self.top_right_grid.addWidget(self.save_model_name_txt, 1, 1)
        
        self.output_count_txt = QtGui.QLabel()
        self.top_right_grid.addWidget(self.output_count_txt, 3,0,1,4)

        #Saves the extremely random decision tree model.
        self.save_model_desc_txt = QtGui.QLineEdit('Insert Model Description')
        self.save_model_desc_txt.setFixedWidth(200)
        self.top_right_grid.addWidget(self.save_model_desc_txt, 2, 0, 1, 4)
        self.clear_dots_btn = QtGui.QPushButton('Clear All ROI')
        self.top_right_grid.addWidget(self.clear_dots_btn, 1, 2)

        #Shows the kernel label distributions
        self.kernel_show_btn = QtGui.QPushButton('Show Kernel')
        self.clear_dots_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.kernel_show_btn, 2, 2)

        self.top_right_grid.setRowStretch(4,2)

        #Sets up the image panel splitter.
        image_panel = QtGui.QSplitter(QtCore.Qt.Horizontal)
        image_panel.addWidget(self.canvas1)
        image_panel.addWidget(self.canvas2)

        #Sets up mouse settings on the image
        self.canvas1.mpl_connect('axes_enter_event', self.on_enter)
        self.canvas1.mpl_connect('axes_leave_event', self.on_leave)
        self.bpe =self.canvas1.mpl_connect('button_press_event', self.on_click)
        self.bre =self.canvas1.mpl_connect('button_release_event', self.on_unclick)
        self.ome =self.canvas1.mpl_connect('motion_notify_event', self.on_motion)

        #Splitter which separates the controls at the top and the images below.
        splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        hbox1 = QtGui.QWidget()
        hbox1.setLayout(top_panel)
        splitter.addWidget(hbox1)
        splitter.addWidget(image_panel)
        box.addWidget(splitter)

        #Status bar which is located beneath images.
        self.image_status_text = QtGui.QStatusBar()
        box.addWidget(self.image_status_text)
        self.image_status_text.showMessage('Status: Please Select a Region and Click \'Save ROI\'. ')
        
        #Connects the buttons.
        self.save_ROI_btn.clicked.connect(self.save_roi_fn)
        self.save_dots_btn.clicked.connect(self.save_dots_fn)
        self.prev_im_btn.clicked.connect(self.prev_im_btn_fn)
        self.next_im_btn.clicked.connect(self.next_im_btn_fn)
        self.sel_ROI_btn.clicked.connect(self.sel_ROI_btn_fn)
        self.remove_dots_btn.clicked.connect(self.remove_dots_btn_fn)
        self.train_model_btn.clicked.connect(self.train_model_btn_fn)
        #self.feat_scale_change_btn.clicked.connect(self.feat_scale_change_btn_fn)
        self.kernel_show_btn.clicked.connect(self.kernel_btn_fn)
        self.clear_dots_btn.clicked.connect(self.clear_dots_fn)
        self.save_model_btn.clicked.connect(self.saveForestFn)
        
        #Initialises the variables for the beginning of the counting.
        par_obj.first_time = True
        par_obj.dots = []
        par_obj.rects = np.zeros((1,4))
        par_obj.var =[]
        par_obj.saved_dots =[]
        par_obj.saved_ROI =[]
        par_obj.subdivide_ROI=[]
        self.m_Cursor = self.makeCursor()
    def report_progress(self,message):
        self.image_status_text.showMessage('Status: '+message)
        app.processEvents()
    def keyPressEvent(self, ev):
        """When the . and , keys are pressed"""
        if ev.key() == QtCore.Qt.Key_Period:
            self.next_im_btn_fn()
        if ev.key() == QtCore.Qt.Key_Comma:
            self.prev_im_btn_fn()
    def wheelEvent(self, event):
        """When the mousewheel is rotated"""
        if event.delta() > 0:
            self.next_im_btn_fn()
        if event.delta() < 0:
            self.prev_im_btn_fn()
    def loadTrainFn(self):
        #Win_fn()
        channel_wid = QtGui.QWidget()
        channel_lay = QtGui.QHBoxLayout()
        for b in range(0,par_obj.numCH):    
            name = 'self.CH_cbx'+str(b)+'= checkBoxCH()'
            exec(name)
            name = 'self.CH_cbx'+str(b)+'.setText(\'CH '+str(b+1)+'\')'
            exec(name)  
            name = 'self.CH_cbx'+str(b)+'.setChecked(True)'
            exec(name)
            name = 'self.CH_cbx'+str(b)+'.type=\'visual_ch\''
            exec(name)
            name = 'self.CH_cbx'+str(b)+'.id='+str(b)
            exec(name)
            name = 'channel_lay.addWidget(self.CH_cbx'+str(b)+')';
            exec(name)
        channel_lay.addStretch()
        channel_wid.setLayout(channel_lay)
        win.top_left_grid.addWidget(channel_wid,1,0,1,3)
        win_tab.setCurrentWidget(win)
        app.processEvents()
        self.checkChange()

    def on_click(self,event):
        """When the image is clicked"""
        par_obj.mouse_down = True
        
        #When the draw ROI functionality is enabled:
        if(par_obj.draw_ROI == True):
            
            self.x1 = event.xdata
            self.y1 = event.ydata
            par_obj.ori_x = event.xdata
            par_obj.ori_y = event.ydata
            
            


    def on_motion(self,event):
        """When the mouse is being dragged"""
        #When the draw ROI functionality is enabled:
        if(par_obj.draw_ROI == True and par_obj.mouse_down == True):
            #Finds current cursor position
            
            par_obj.ori_x_2 = event.xdata
            par_obj.ori_y_2 = event.ydata
            try:
                
                    
                    self.plt1.lines.remove(self.l1[0])
                    self.plt1.lines.remove(self.l2[0])
                    self.plt1.lines.remove(self.l3[0])
                    self.plt1.lines.remove(self.l4[0])
                    
                
            except:
                pass
            self.plt1.autoscale(False)
            self.l1 = self.plt1.plot([self.x1, event.xdata], [self.y1, self.y1], '-' ,color='r')
            self.l2 = self.plt1.plot([event.xdata, event.xdata], [self.y1, event.ydata], '-' ,color='r')
            self.l3 = self.plt1.plot([event.xdata, self.x1], [ event.ydata,  event.ydata], '-' ,color='r')
            self.l4 = self.plt1.plot([self.x1, self.x1], [ event.ydata, self.y1], '-' ,color='r')
            
            
            #self.plt1.Line2D([event.xdata, event.xdata], [self.y1, event.ydata], transform=self.plt1.transData,  figure=self.plt1,color='r')
            #self.plt1.Line2D([event.xdata, self.x1], [ event.ydata,  event.ydata], transform=self.plt1.transData,  figure=self.plt1,color='r')
            #self.plt1.Line2D([self.x1, self.x1], [ event.ydata, self.y1], transform=self.plt1.transData,  figure=self.plt1,color='r')
            
            self.canvas1.draw()
            
            
    def on_unclick(self, event):
        """When the mouse is released"""
        par_obj.mouse_down = False
        
        #If we are in the roi drawing phase
        if(par_obj.draw_ROI == True):
            t2 = time.time()
            x = event.xdata
            y = event.ydata
            
            par_obj.rect_w = x - par_obj.ori_x
            par_obj.rect_h = y - par_obj.ori_y
            
            #Corrects the corrdinates if out of rectangle.
            if(x < 0): x=0
            if(y < 0): y=0
            if(x > par_obj.width): x=par_obj.width-1
            if(y > par_obj.height): y=par_obj.height-1
            t1 = time.time()
            print t1-t2
        #If we are in the dot drawing phase
        if(par_obj.draw_dots == True):   
            x = int(np.round(event.xdata,0))
            y = int(np.round(event.ydata,0))
        
            #Are we with an existing box.
            if(x > par_obj.rects[1]-par_obj.roi_tolerance and x < (par_obj.rects[1]+ par_obj.rects[3])+par_obj.roi_tolerance and y > par_obj.rects[2]-par_obj.roi_tolerance and y < (par_obj.rects[2]+ par_obj.rects[4])+par_obj.roi_tolerance):
                appendDot = True
                #Appends dots to array if in an empty pixel.
                if(par_obj.dots.__len__()>0):
                    for i in range(0,par_obj.dots.__len__()):
                        if (x == par_obj.dots[i][1] and y == par_obj.dots[i][2]):
                            appendDot = False                            

                if(appendDot == True):
                    par_obj.dots.append((par_obj.curr_img,x,y))
                    i = par_obj.dots[-1]
                    self.plt1.autoscale(False)
                    self.plt1.plot([i[1]-5,i[1]+5],[i[2],i[2]],'-',color='r')
                    self.plt1.plot([i[1],i[1]],[i[2]-5,i[2]+5],'-',color='r')
                    self.canvas1.draw()

        if(par_obj.remove_dots == True): 
            #par_obj.pixMap = QtGui.QPixmap(q2r.rgb2qimage(par_obj.imgs[par_obj.curr_img]))
            x = event.xdata
            y = event.ydata
            self.draw_saved_dots_and_roi()
            #Are we with an existing box.
            if(x > par_obj.rects[1]-par_obj.roi_tolerance and x < (par_obj.rects[1]+ par_obj.rects[3])+par_obj.roi_tolerance and y > par_obj.rects[2]-par_obj.roi_tolerance and y < (par_obj.rects[2]+ par_obj.rects[4])+par_obj.roi_tolerance):
                #Appends dots to array if in an empty pixel.
                if(par_obj.dots.__len__()>0):
                    for i in range(0,par_obj.dots.__len__()):
                        if ((abs(x -par_obj.dots[i][1])<3 and abs(y - par_obj.dots[i][2])<3)):
                            par_obj.dots.pop(i)
                            par_obj.saved_dots.append(par_obj.dots)
                            par_obj.saved_ROI.append(par_obj.rects)
                            self.update_density_fn()
                            par_obj.dots = par_obj.saved_dots[par_obj.ROI_index[par_obj.roi_select]]
                            par_obj.rects =  par_obj.saved_ROI[par_obj.ROI_index[par_obj.roi_select]]
                            par_obj.saved_dots.pop(par_obj.ROI_index[par_obj.roi_select])
                            par_obj.saved_ROI.pop(par_obj.ROI_index[par_obj.roi_select])
                            break 
            

            for i in range(0,self.plt1.lines.__len__()):
                self.plt1.lines.pop(0)
            

            self.dots_and_square(par_obj.dots,par_obj.rects,'y')
            print 'here1'
            self.canvas1.draw()
            print 'here2'
        if(par_obj.select_ROI== True):
            x = event.xdata
            y = event.ydata
            for b in range(0,par_obj.ROI_index.__len__()):
                dots = par_obj.saved_dots[par_obj.ROI_index[b]]
                rects = par_obj.saved_ROI[par_obj.ROI_index[b]]
                if(x > rects[1] and x < (rects[1]+ rects[3]) and y > rects[2] and y < (rects[2]+ rects[4])):
                    
                    par_obj.roi_select = b
                    par_obj.dots = par_obj.saved_dots[par_obj.ROI_index[par_obj.roi_select]]
                    par_obj.rects =  par_obj.saved_ROI[par_obj.ROI_index[par_obj.roi_select]]
                    par_obj.saved_dots.pop(par_obj.ROI_index[par_obj.roi_select])
                    par_obj.saved_ROI.pop(par_obj.ROI_index[par_obj.roi_select])
                   
                    for i in range(0,self.plt1.lines.__len__()):
                        self.plt1.lines.pop(0)
                    self.draw_saved_dots_and_roi()
                    self.dots_and_square(dots,rects,'y')
                    self.canvas1.draw()
                    self.sel_ROI_btn.setEnabled(False)
                    self.save_dots_btn.setEnabled(True)
                    self.remove_dots_btn.setEnabled(True)
                    par_obj.select_ROI= False
                    par_obj.draw_ROI = False
                    par_obj.draw_dots = True
    def dots_and_square(self, dots,rects,colour):
       
        
        #self.l5 = lines.Line2D([rects[1], rects[1]+rects[3]], [rects[2],rects[2]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.l6 = lines.Line2D([rects[1]+rects[3], rects[1]+rects[3]], [rects[2],rects[2]+rects[4]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.l7 = lines.Line2D([rects[1]+rects[3], rects[1]], [rects[2]+rects[4],rects[2]+rects[4]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.l8 = lines.Line2D([rects[1], rects[1]], [rects[2]+rects[4],rects[2]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.plt1.lines.extend([self.l5,self.l6,self.l7,self.l8])
        self.plt1.autoscale(False)
        self.plt1.plot([rects[1], rects[1]+rects[3]], [rects[2],rects[2]], '-',color=colour)
        self.plt1.plot([rects[1]+rects[3], rects[1]+rects[3]], [rects[2],rects[2]+rects[4]], '-',color=colour)
        self.plt1.plot([rects[1]+rects[3], rects[1]], [rects[2]+rects[4],rects[2]+rects[4]], '-',  figure=self.plt1,color=colour)
        self.plt1.plot([rects[1], rects[1]], [rects[2]+rects[4],rects[2]], '-',  figure=self.plt1,color=colour)
        

            #Draws dots in list
        for i in iter(dots):
            self.plt1.plot([i[1]-5,i[1]+5],[i[2],i[2]],'-',color=colour)
            self.plt1.plot([i[1],i[1]],[i[2]-5,i[2]+5],'-',color=colour)
            
        
        
    
        return
    def makeCursor(self):
        m_LPixmap = QtGui.QPixmap(28, 28)
        bck = QtGui.QColor(168, 34, 3)
        bck.setAlpha(0)
        m_LPixmap.fill(bck)
        qp = QtGui.QPainter(m_LPixmap)
        qp.setPen(QtGui.QColor(0, 255, 0,200))
        qp.drawLine(14,0,14,28)
        qp.drawLine(0,14,28,14)
        qp.setOpacity(1.0)
        m_Cursor = QtGui.QCursor(m_LPixmap)
        qp.setOpacity(0.0)
        qp.end()
        return m_Cursor
    def on_enter(self,ev):
        #Changes cursor to the special crosshair on entering image pane.
        QtGui.QApplication.setOverrideCursor(self.m_Cursor)
    def on_leave(self,ev):
        QtGui.QApplication.restoreOverrideCursor()
    def save_roi_fn(self):
        #If there is no width or height either no roi is selected or it is too thin.        
        success = v2.save_roi_fn(par_obj)
        if success == True:
            print ('Saved ROI')
            win.image_status_text.showMessage('Status: Select instances in region then click \'save Dots\' ')
            par_obj.draw_ROI = False
            par_obj.draw_dots = True
            win.save_ROI_btn.setEnabled(False)
            win.save_dots_btn.setEnabled(True)
            win.remove_dots_btn.setEnabled(True)
            win.sel_ROI_btn.setEnabled(False)
            par_obj.remove_dots = False
    def deleteDotsFn(self,sel_ROI_btn_fn):
        print('Dot deleted')
        par_obj.saved_dots.append(par_obj.dots)
        par_obj.saved_ROI.append(par_obj.rects)
        par_obj.dots = par_obj.saved_dots[par_obj.ROI_index[par_obj.roi_select]]
        par_obj.rects = par_obj.saved_ROI[par_obj.ROI_index[par_obj.roi_select]]
        par_obj.saved_dots.pop(par_obj.ROI_index[par_obj.roi_select])
        par_obj.saved_ROI.pop(par_obj.ROI_index[par_obj.roi_select])
        #Creates the qpainter object
          
       
        #Now we update a density image of the current Image.
        self.update_density_fn()
    def save_dots_fn(self):
        print('Saved Dots')
        win.image_status_text.showMessage('Status: Highlight new ROI or train. ')
        win.train_model_btn.setEnabled(True)
        par_obj.saved_dots.append(par_obj.dots)
        par_obj.saved_ROI.append(par_obj.rects)
        self.draw_saved_dots_and_roi()
        self.save_ROI_btn.setEnabled(True)
        self.save_dots_btn.setEnabled(False)
        self.remove_dots_btn.setEnabled(False)
        self.sel_ROI_btn.setEnabled(True)
        self.clear_dots_btn.setEnabled(True)
        par_obj.draw_ROI = True
        par_obj.draw_dots = False
        par_obj.remove_dots = False
        par_obj.dots_past = par_obj.dots
        par_obj.dots = []
        par_obj.rects = np.zeros((1,4))
        par_obj.ori_x=0
        par_obj.ori_y=0
        par_obj.rect_w=0
        par_obj.rect_h =0
        
        
        self.goto_img_fn(par_obj.curr_img)

        #Now we update a density image of the current Image.
        self.update_density_fn()
    def update_density_fn(self):
        #Construct empty array for current image.
        par_obj.im_for_train = [par_obj.curr_img]
        v2.update_density_fn(par_obj)
        self.plt2.cla()
        self.plt2.imshow(par_obj.dense_array[par_obj.curr_img])
        self.plt2.set_xticklabels([])
        self.plt2.set_yticklabels([])
        self.canvas2.draw()
    def draw_saved_dots_and_roi(self):
        for i in range(0,par_obj.subdivide_ROI.__len__()):
            if(par_obj.subdivide_ROI[i][0] ==par_obj.curr_img):
                rects =par_obj.subdivide_ROI[i]
                dots = []
                self.dots_and_square(dots,rects,'w')
        for i in range(0,par_obj.saved_dots.__len__()):
            if(par_obj.saved_ROI[i][0] == par_obj.curr_img):
                dots = par_obj.saved_dots[i]
                rects = par_obj.saved_ROI[i]
                self.dots_and_square(dots,rects,'w')
        
    def prev_im_btn_fn(self):
        im_num = par_obj.curr_img - 1
        if im_num >-1:
            par_obj.curr_img = im_num
            self.goto_img_fn(im_num)
            
    def next_im_btn_fn(self):
        im_num = par_obj.curr_img + 1
        if im_num <par_obj.test_im_end:
            par_obj.curr_img = im_num
            self.goto_img_fn(im_num)


    def goto_img_fn(self,im_num):
        #Goto and evaluate image function.
        v2.eval_goto_img_fn(im_num,par_obj,self)
        self.draw_saved_dots_and_roi()
        par_obj.dots = []
        par_obj.rects = np.zeros((1,4))
        par_obj.select_ROI= False
        par_obj.draw_ROI = True
        par_obj.draw_dots = False
        par_obj.remove_dots = False
        self.save_ROI_btn.setEnabled(True)
        self.save_dots_btn.setEnabled(False)
        self.remove_dots_btn.setEnabled(False)
        self.sel_ROI_btn.setEnabled(True)
        par_obj.ROI_index=[]
            
            
    def sel_ROI_btn_fn(self):
        par_obj.ROI_index =[]
        if(par_obj.select_ROI== False):
            
            self.save_ROI_btn.setEnabled(False)
            par_obj.select_ROI= True
            par_obj.draw_ROI = False
            par_obj.draw_dots = False
            par_obj.remove_dots = False
            for i in range(0,par_obj.saved_ROI.__len__()):
                if(par_obj.saved_ROI[i][0] == par_obj.curr_img):
                    par_obj.ROI_index.append(i)
            for b in range(0,par_obj.ROI_index.__len__()):
                dots = par_obj.saved_dots[par_obj.ROI_index[b]]
                rects = par_obj.saved_ROI[par_obj.ROI_index[b]]
                self.dots_and_square(dots,rects,'y')
        else:
            self.save_ROI_btn.setEnabled(True)
            par_obj.select_ROI= False
            par_obj.draw_ROI = True
            self.draw_saved_dots_and_roi()
    def remove_dots_btn_fn(self):
        if(par_obj.remove_dots == False):
            par_obj.remove_dots = True
            par_obj.draw_dots = False
        else:
            par_obj.remove_dots = False
            par_obj.draw_dots = True
    def clear_dots_fn(self):
        par_obj.saved_dots = []
        par_obj.saved_ROI = []
        par_obj.dense_array={}
        self.goto_img_fn(par_obj.curr_img)
        self.update_density_fn()
        self.train_model_btn.setEnabled(False)
        self.clear_dots_btn.setEnabled(False)
    def train_model_btn_fn(self):
        self.image_status_text.showMessage('Training Ensemble of Decision Trees. ')
        v2.update_training_samples_fn(par_obj,0)
        self.image_status_text.showMessage('Evaluating Images with the Trained Model. ')
        app.processEvents()    
        v2.evaluate_forest(par_obj,self, False,0)
        v2.make_correction(par_obj, 0)
        self.image_status_text.showMessage('Model Trained. Continue adding samples, or click \'Save Training Model\'. ')
        par_obj.eval_load_im_win_eval = True
        self.goto_img_fn(par_obj.curr_img)
        self.save_model_btn.setEnabled(True)  
    def sigmaOnChange(self,text):
        par_obj.sigma_data = float(text)
        self.update_density_fn()
    def feature_scale_change(self,text):
        par_obj.feature_scale = float(text)
    def feat_scale_change_btn_fn(self):
        self.feat_scale_change_btn.setEnabled(False)
        print('Training Features')
        processImgs()
        v2.update_density_fn()
        self.feat_scale_change_btn.setEnabled(True)
        self.image_status_text.showMessage('Model Trained. Continue adding samples, or click \'Save Training Model\'. ')  
        self.save_model_btn.setEnabled(True)  
        v2.eval_pred_show_fn(par_obj.curr_img,par_obj,self)

    def kernel_btn_fn(self):
        """Shows the kernels on the image."""
        if par_obj.kernel_toggle == False:
            par_obj.kernel_toggle = True
            self.kernel_show_btn.setText('Show Prediction')
            self.update_density_fn()
        else:
            par_obj.kernel_toggle = False
            self.kernel_show_btn.setText('Show Kernel')
            self.predShowFn()

    def predShowFn(self):
        #Captures the button event.
        v2.eval_pred_show_fn(par_obj.curr_img,par_obj,self)

    def saveForestFn(self):
        
        path = os.path.expanduser('~')+'/.densitycount/models/'
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        local_time = time.asctime( time.localtime(time.time()) )   
        par_obj.modelName = str(self.save_model_name_txt.text())
        par_obj.modelDescription = str(self.save_model_desc_txt.text())
        
        cleanString = re.sub('\W+', '', par_obj.modelName )

        basename = path+"pv20"
        suffix = str(int(round(time.time(),0)))
        filename = "_".join([basename,suffix,str(cleanString),  ".mdla"])
        save_file = {}

        #Formats image to make a better icon.
        if par_obj.save_im.shape[0] > 300 and  par_obj.save_im.shape[1] > 300:
            save_im = np.zeros((300, 300, 3))
            cent_y = np.floor(par_obj.save_im.shape[0]/2).astype(np.int32)
            cent_x = np.floor(par_obj.save_im.shape[1]/2).astype(np.int32)
            if par_obj.save_im.shape[2]> 2:
                save_im[:,:,0] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,2]
                save_im[:,:,1] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,1]
                save_im[:,:,2] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,0]
            else:
                save_im[:,:,0] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,0]
                save_im[:,:,1] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,0]
                save_im[:,:,2] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,0]
        else:
            save_im = np.zeros((par_obj.save_im.shape[0], par_obj.save_im.shape[1], 3))
            if par_obj.save_im.shape[2]> 2:
                save_im[:,:,0] = par_obj.save_im[:, :, 2]
                save_im[:,:,1] = par_obj.save_im[:, :, 1]
                save_im[:,:,2] = par_obj.save_im[:, :, 0]
            else:
                save_im[:,:,0] = par_obj.save_im[:, :,0]
                save_im[:,:,1] = par_obj.save_im[:, :,0]
                save_im[:,:,2] = par_obj.save_im[:, :,0]
        
        save_file ={"name":par_obj.modelName,'description':par_obj.modelDescription,"c":par_obj.c,"M":par_obj.M,\
        "sigma_data":par_obj.sigma_data, "model":par_obj.RF, "date":local_time, "feature_type":par_obj.feature_type, \
        "feature_scale":par_obj.feature_scale, "ch_active":par_obj.ch_active, "limit_ratio_size":par_obj.limit_ratio_size, \
        "max_depth":par_obj.max_depth, "min_samples":par_obj.min_samples_split, "min_samples_leaf":par_obj.min_samples_leaf,\
        "max_features":par_obj.max_features, "num_of_tree":par_obj.num_of_tree, "file_ext":par_obj.file_ext, "imFile":save_im,\
        "gt_vec":par_obj.gt_vec, "error_vec":par_obj.error_vec};
        
        pickle.dump(save_file, open(filename, "wb"))
        self.save_model_btn.setEnabled(False)
        self.report_progress('Model Saved.')
        
    def checkChange(self):
        v2.eval_goto_img_fn(par_obj.curr_img, par_obj, self)

class checkBoxCH(QtGui.QCheckBox):
    def __init__(self):
        QtGui.QCheckBox.__init__(self)
        self.stateChanged.connect(self.stateChange)
        self.type = None;
    def stateChange(self):
        if self.type == 'feature_ch':
            par_obj.ch_active = []
            for i in range(0, par_obj.numCH):
                name = 'v = loadWin.CH_cbx'+str(i+1)+'.checkState()'
                exec(name)
                if v == 2:
                    par_obj.ch_active.append(i)

            newImg = np.zeros((par_obj.height, par_obj.width, 3))
            if par_obj.ch_active.__len__() > 1:
                 
                 for b in par_obj.ch_active:
                    newImg[:, :, b] = par_obj.ex_img[:, :, b]

            elif par_obj.ch_active.__len__() ==1:
                newImg = par_obj.ex_img[:, :, par_obj.ch_active[0]]
            loadWin.plt1.cla()
            loadWin.plt1.imshow(255-newImg)
            loadWin.draw_saved_dots_and_roi()
            loadWin.plt1.set_xticklabels([])
            loadWin.plt1.set_yticklabels([])
            loadWin.canvas1.draw()
        if self.type == 'visual_ch':
            #print 'visualisation changed channel'
            win.goto_img_fn(par_obj.curr_img)

        
class parameterClass:
    def __init__(self):
      
        self.roi_tolerance = 10
        
        #Parameters of sampling
        self.limit_sample = True
        self.limit_ratio = True #whether to use ratio of roi pixels
        self.limit_ratio_size =21 #Gives 3000 patches for 255*255 image.
        self.limit_size = 3000 #patches per image or ROI.
        #Random Forest parameters
        self.pw = 1
        self.max_depth=10
        self.min_samples_split=20 
        self.min_samples_leaf=10  
        self.max_features = 7
        self.num_of_tree = 30
        self.feature_scale = 0.8
        self.x_limit = 5024
        self.y_limit = 5024
        self.sigma_data = 1.0
        self.p_size = 1
        self.fresh_features = True
        self.crop_x2 = 0
        self.crop_x1 =0
        self.curr_img = 0
        self.numCH =0;
        self.RF ={}
        self.dense_array={}
        self.frames_2_load ={}
        #Auto mode.
        self.auto = True
        self.draw_ROI = True
        self.remove_dots = False
        self.draw_dots =False
        self.select_ROI= False
        self.mouse_down = False
        self.kernel_toggle = False
        self.eval_load_im_win_eval = False
        self.ex_img = []


#generate layout
app = QtGui.QApplication([])

# Create and display the splash screen
splash_pix = QtGui.QPixmap('splash_loading.png')
splash = QtGui.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
splash.setMask(splash_pix.mask())
splash.show()
app.processEvents()
#Creates tab widget.
win_tab = QtGui.QTabWidget()
#Creates win, an instance of QWidget
par_obj  = parameterClass()  
win = Win_fn(par_obj)
loadWin= Load_win_fn(par_obj,win)

#Adds win tab and places button in par_obj.
win_tab.addTab(loadWin, "Load Images")
win_tab.addTab(win, "Train Model")

#Defines size of the widget.
win_tab.resize(1000,600)
time.sleep(2.0)
splash.finish(win_tab)
win_tab.show()

# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
