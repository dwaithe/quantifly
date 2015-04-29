#Experimental interface
import pylab
from PyQt4 import QtGui, QtCore, Qt, QtWebKit
import time

import vigra
import scipy
from scipy.ndimage import filters
from scipy.sparse.csgraph import _validation
from sklearn.ensemble import ExtraTreesRegressor
import thread
import random
import csv
import cPickle as pickle
import datetime
import errno
import os
import numpy as np

import datetime
from scipy.special import _ufuncs_cxx
import sklearn.utils.lgamma
from gnu import return_license

#Apparently matplotlib slows the loading drammatically due to a font cache issue. This resolves it.
try:
    #mac location.
    os.remove(os.path.expanduser('~')+'/.matplotlib/fontList.cache')
    print 'removing matplotlib cache'
except:
    pass
try:
    #Alternate location.
    os.remove(os.path.expanduser('~')+'/.cache/matplotlib')
    print 'removing matplotlib cache'
except:
    pass

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
   

class Eval_load_im_win(QtGui.QWidget):
    def __init__(self,par_obj):
        super(Eval_load_im_win, self).__init__()
        """Setups the load image interface"""

        #Stops reinitalise of text which can produce buggy text.
        #if par_obj.evalLoadImWin_loaded !=True:
        vbox0 = QtGui.QVBoxLayout()
        self.setLayout(vbox0)


        
       

        #Load images button
        self.loadImages_button_panel = QtGui.QHBoxLayout()
        self.loadImages_button = QtGui.QPushButton("Add Images")
        self.loadImages_button.move(20, 20)
        self.loadImages_button_panel.addWidget(self.loadImages_button)
        self.loadImages_button_panel.addStretch()

        about_btn = QtGui.QPushButton('About')
        about_btn.clicked.connect(self.on_about)
        self.loadImages_button_panel.addWidget(about_btn)


        #Table widget which displays
        self.modelTabIm_panel = QtGui.QHBoxLayout()
        self.modelTabIm = QtGui.QTableWidget()
        self.modelTabIm.setRowCount(1)
        self.modelTabIm.setColumnCount(3)
        self.modelTabIm.setColumnWidth(0,125)
        self.modelTabIm.resize(550,200)
        self.modelTabIm.setHorizontalHeaderLabels(QtCore.QString(",Image name, Path").split(","))
        self.modelTabIm.hide()
        self.modelTabIm.setEditTriggers( QtGui.QTableWidget.NoEditTriggers )
        self.modelTabIm_panel.addWidget(self.modelTabIm)
        self.modelTabIm_panel.addStretch()

        #Starts file dialog calss
        self.imLoad = File_Dialog(par_obj,self,self.modelTabIm)
        self.imLoad.type = 'im'
        self.loadImages_button.clicked.connect(self.imLoad.showDialog)
        
        vbox0.addLayout(self.loadImages_button_panel)
        vbox0.addLayout(self.modelTabIm_panel)
        vbox0.addStretch()
        
        #Move to training button.
        self.selIntButton_panel = QtGui.QHBoxLayout()
        self.selIntButton = QtGui.QPushButton("Goto Model Import")
        self.selIntButton.clicked.connect(self.goto_model_import)
        self.selIntButton.move(20, 520)
        self.selIntButton.setEnabled(False)
        self.selIntButton_panel.addWidget(self.selIntButton)
        self.selIntButton_panel.addStretch()
        vbox0.addLayout(self.selIntButton_panel)
        

        #Status bar to report whats going on.
        
        self.image_status_text = QtGui.QStatusBar()
        self.image_status_text.showMessage('Status: Highlight training images in folder. ')
        vbox0.addWidget(self.image_status_text)
    def on_about(self):
        self.about_win = QtGui.QWidget()
        self.about_win.setWindowTitle('About QuantiFly Software v2.0')
        
        license = return_license()
        #with open ('GNU GENERAL PUBLIC LICENSE.txt', "r") as myfile:
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
                '''+str(license)[2:-2]+'''
                
                
              </form>
            </body>
          </html>
        ''')
        layout.addWidget(self.view)
        self.about_win.setLayout(layout)
        self.about_win.show()
        self.about_win.raise_()   
    def goto_model_import(self):
            win_tab.setCurrentWidget(evalLoadModelWin)
            #Checks the correct things are disabled.
            evalLoadModelWin.gotoEvalButton.setEnabled(False)
            evalImWin.prev_im_btn.setEnabled(False)
            evalImWin.next_im_btn.setEnabled(False)
            evalImWin.eval_im_btn.setEnabled(False)
            par_obj.eval_load_im_win_eval = False

class Eval_load_model_win(QtGui.QWidget):
    """Interface which allows selection of model."""
    def __init__(self,par_obj):
        
        super(Eval_load_model_win,self).__init__()
        
        #The main layout
        box = QtGui.QVBoxLayout()
        self.setLayout(box)
        
        hbox0 = QtGui.QHBoxLayout()
        box.addLayout(hbox0)
        #The two principle columns
        vbox0 = QtGui.QVBoxLayout()
        vbox1 = QtGui.QVBoxLayout()
        hbox0.addLayout(vbox0)
        hbox0.addLayout(vbox1)

        #Display available models.
        #Find all files in folder with serielize prefix.
        
        files = os.listdir(par_obj.forPath)
        filesRF =[]
        for b in range(0,files.__len__()):
            
            if(os.path.splitext(files[b])[1] == '.pkl'):
                filesRF.append(os.path.splitext(files[b])[0] )
            if(os.path.splitext(files[b])[1] == '.mdla'):
                filesRF.append(os.path.splitext(files[b])[0] )
        

        filesLen =filesRF.__len__()

      
        self.modelTabFor = QtGui.QTableWidget()
        self.modelTabFor.setRowCount(1)
        self.modelTabFor.setColumnCount(3)
        self.modelTabFor.setColumnWidth(2,200)
        self.modelTabFor.resize(600,500)
        self.modelTabFor.setHorizontalHeaderLabels(QtCore.QString(",model name, date and time saved").split(","))
        self.modelTabFor.setEditTriggers( QtGui.QTableWidget.NoEditTriggers )

        vbox0.addWidget(self.modelTabFor)

        #Button for going to model evaluation.
        self.gotoEvalButton = QtGui.QPushButton("Goto Image Evaluation")
        self.gotoEvalButton.setEnabled(False)
        self.gotoEvalButton.clicked.connect(self.process_imgs)
        
        vbox0.addWidget(self.gotoEvalButton)

        #Status text.
        self.image_status_text = QtGui.QStatusBar()
        self.image_status_text.resize(300,20)
        self.image_status_text.setStyleSheet("QLabel {  color : green }")
        self.image_status_text.showMessage('Status: Please click a model from above and then click \'Load Model\'. ')

        #The second column.
        self.modelIm_panel = QtGui.QHBoxLayout()

        self.figure1 = Figure(figsize=(3, 3), dpi=100)
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure1.patch.set_facecolor('grey')
        
        
        self.plt1 = self.figure1.add_subplot(1, 1, 1)
        im_RGB = np.zeros((300, 300))
        #Makes sure it spans the whole figure.
        self.figure1.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)
        self.plt1.imshow(im_RGB)

        #Removes the tick labels
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])

        vbox1.addWidget(self.canvas1)
        


        self.modelImTxt1_panel = QtGui.QHBoxLayout()
        self.modelImTxt2_panel = QtGui.QHBoxLayout()
        self.modelImTxt3_panel = QtGui.QHBoxLayout()
        self.modelImTxt4_panel = QtGui.QHBoxLayout()
        self.modelImTxt5_panel = QtGui.QHBoxLayout()
        self.modelImTxt6_panel = QtGui.QHBoxLayout()

        self.modelImTxt1 = QtGui.QLabel()
        self.modelImTxt2 = QtGui.QLabel()
        self.modelImTxt3 = QtGui.QLabel()
        self.modelImTxt4 = QtGui.QLabel()
        self.modelImTxt5 = QtGui.QLabel()
        self.modelImTxt6 = QtGui.QLabel()

        self.modelImTxt1.setText('Name: ')
        self.modelImTxt1.resize(400,25)
        self.modelImTxt1_panel.addWidget(self.modelImTxt1)
        self.modelImTxt1_panel.addStretch()
        
        self.modelImTxt2.setText('Description: ')
        self.modelImTxt2.resize(400,25)
        self.modelImTxt2_panel.addWidget(self.modelImTxt2)
        self.modelImTxt2_panel.addStretch()
        
        self.modelImTxt3.setText('Sigma Data: ')
        self.modelImTxt3.resize(400,25)
        self.modelImTxt3_panel.addWidget(self.modelImTxt3)
        self.modelImTxt3_panel.addStretch()
        
        self.modelImTxt4.setText('Feature Scale: ')
        self.modelImTxt4.resize(400,25)
        self.modelImTxt4_panel.addWidget(self.modelImTxt4)
        self.modelImTxt4_panel.addStretch()
        
        self.modelImTxt5.setText('Feature Type: ')
        self.modelImTxt5.resize(400,25)
        self.modelImTxt5_panel.addWidget(self.modelImTxt5)
        self.modelImTxt5_panel.addStretch()
        
        self.modelImTxt6.setText('Channels: ')
        self.modelImTxt6.resize(400,25)
        self.modelImTxt6_panel.addWidget(self.modelImTxt6)
        self.modelImTxt6_panel.addStretch()

        vbox1.addLayout(self.modelImTxt1_panel)
        vbox1.addLayout(self.modelImTxt2_panel)
        vbox1.addLayout(self.modelImTxt3_panel)
        vbox1.addLayout(self.modelImTxt4_panel)
        vbox1.addLayout(self.modelImTxt5_panel)
        vbox1.addLayout(self.modelImTxt6_panel)
        vbox1.addStretch()

        
        
        
        
        c =0
        for i in range(0,filesRF.__len__()):
                
                strFn = filesRF[i].split('_')
                
                
                
                if(str(strFn[0]) == 'pv1.3'):


                    self.modelTabFor.setRowCount(c+1)
                    btn = loadModelBtn(par_obj,self,self.modelTabFor,i,filesRF[i])

                    btn.setText('Click to View')
                    self.modelTabFor.setCellWidget(c, 0, btn)

                    text1 = QtGui.QLabel(self.modelTabFor)
                    text1.setText(str(' '+strFn[1]))
                    self.modelTabFor.setCellWidget(c,1,text1)

                    text2 = QtGui.QLabel(self.modelTabFor)
                    text2.setText(str(' '+strFn[2]))
                    self.modelTabFor.setCellWidget(c,2,text2)
                    c= c+1
                if str(strFn[0]) == 'pv20':
                    

                    self.modelTabFor.setRowCount(c+1)
                    btn = loadModelBtn(par_obj,self,self.modelTabFor,i,filesRF[i])

                    btn.setText('Click to View')
                    self.modelTabFor.setCellWidget(c, 0, btn)

                    text1 = QtGui.QLabel(self.modelTabFor)
                    text1.setText(str(' '+strFn[2]))
                    self.modelTabFor.setCellWidget(c,1,text1)

                    text2 = QtGui.QLabel(self.modelTabFor)
                    text2.setText(str(' '+datetime.datetime.fromtimestamp(float(strFn[1])).strftime('%Y-%m-%d %H:%M:%S')))
                    self.modelTabFor.setCellWidget(c,2,text2)
                    c= c+1

        box.addWidget(self.image_status_text)
    def loadModelFn(self,par_obj, fileName):
        """Shows details of the model when loaded"""


        par_obj.selectedModel = par_obj.forPath+fileName
        par_obj.evaluated = False
        self.image_status_text.showMessage('Status: Loading previously trained model. ')
        app.processEvents()

        ver = par_obj.selectedModel.split('/')[-1][0:5]
        
        if(ver =='pv20_'):
            save_file = pickle.load(open(par_obj.selectedModel+str('.mdla'), "rb"))
            par_obj.modelName = save_file["name"]
            par_obj.modelDescription = save_file["description"]
            par_obj.RF = save_file["model"]
            local_time = save_file["date"]
            par_obj.M = save_file["M"]
            par_obj.c = save_file["c"]
            par_obj.feature_type = save_file["feature_type"]
            par_obj.feature_scale = save_file["feature_scale"]
            par_obj.sigma_data = save_file["sigma_data"]
            par_obj.ch_active = save_file["ch_active"]
            par_obj.limit_ratio_size = save_file["limit_ratio_size"]
            par_obj.max_depth = save_file["max_depth"]
            par_obj.min_samples_split = save_file["min_samples"]
            par_obj.min_samples_leaf = save_file["min_samples_leaf"]
            par_obj.max_features = save_file["max_features"]
            par_obj.num_of_tree = save_file["num_of_tree"]
            par_obj.file_ext = save_file["file_ext"]
            par_obj.gt_vec = save_file["gt_vec"]
            par_obj.error_vec = save_file["error_vec"]
            save_im = 255-save_file["imFile"]
            self.image_status_text.showMessage('Status: Model loaded. ')
            success = True
            

        if(ver =='pv1.3'):
            #Load in parameters from file.
            par_obj.file_ext,par_obj.RF,par_obj.sigma_data,par_obj.feature_scale,par_obj.feature_type,par_obj.ch_active,par_obj.modelName,par_obj.modelDescription = pickle.load(open(par_obj.selectedModel+str('.pkl'), "rb"))
            
            #save_im =cv2.imread(par_obj.selectedModel+"im.png")
            self.image_status_text.showMessage('Status: Model loaded. ')
            #Display details about model.
            success, statusText = v2.import_data_fn(par_obj,par_obj.file_array)

        if success == True:
            self.modelImTxt1.setText('Name: '+str(par_obj.modelName))
            self.modelImTxt2.setText('Description: '+str(par_obj.modelDescription ))
            self.modelImTxt3.setText('Sigma Data: '+str(par_obj.sigma_data))
            self.modelImTxt4.setText('Feature Scale: '+str(par_obj.feature_scale))
            self.modelImTxt5.setText('Feature Type: '+str(par_obj.feature_type))
            self.modelImTxt6.setText('Channels: '+str(par_obj.ch_active))
            self.plt1.imshow(save_im)
            self.canvas1.draw()
            self.gotoEvalButton.setEnabled(True)
            evalImWin.eval_im_btn.setEnabled(True)
            par_obj.eval_load_im_win_eval = False

    def process_imgs(self):
        par_obj.left_2_calc =[]
        par_obj.frames_2_load ={}
        par_obj.feat_arr ={}
        
        
        #Now we commit our options to our imported files.
        if par_obj.file_ext == 'png':
            for i in range(0,par_obj.file_array.__len__()):
                par_obj.left_2_calc.append(i)
                par_obj.frames_2_load[i] = [0]
            
        elif par_obj.file_ext =='tiff' or par_obj.file_ext =='tif':
            if par_obj.tiff_file.maxFrames>1:
                for i in range(0,par_obj.file_array.__len__()):
                    par_obj.left_2_calc.append(i)
                    par_obj.frames_2_load[i] = range(0,par_obj.tiff_file.maxFrames)
                    #try:
                    #    np.array(list(self.hyphen_range(fmStr)))-1
                    #    par_obj.frames_2_load[i] = np.array(list(self.hyphen_range(fmStr)))-1
                    #except:
                    #    self.image_status_text.showMessage('Status: The supplied range of image frames is in the wrong format. Please correct and click confirm images.')
                    #    return
                
               
            else:
                for i in range(0,par_obj.file_array.__len__()):
                    par_obj.left_2_calc.append(i)
                    par_obj.frames_2_load[i] = [0]
                
        count = 0
        for b in par_obj.left_2_calc:
            frames =par_obj.frames_2_load[b]
            for i in frames:
                count = count+1
                
        par_obj.test_im_start= 0
        par_obj.test_im_end= count
        

        win_tab.setCurrentWidget(evalImWin)
        evalImWin.prev_im_btn.setEnabled(True)
        evalImWin.next_im_btn.setEnabled(True)
        
        evalImWin.loadTrainFn()
        v2.eval_goto_img_fn(par_obj.curr_img,par_obj,evalImWin)
class Eval_disp_im_win(QtGui.QWidget):
    """ Arranges widget to visualise the input images and ouput prediction. """
    def __init__(self,par_obj):
        super(Eval_disp_im_win, self).__init__()
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
        
        #Sets up the button which changes to the prev image
        self.prev_im_btn = QtGui.QPushButton('Prev Image')
        self.prev_im_btn.clicked.connect(self.prev_im_btn_fn)
        
        #Sets up the button which changes to the next Image.
        self.next_im_btn = QtGui.QPushButton('Next Image')
        self.next_im_btn.clicked.connect(self.next_im_btn_fn)
        
        
        #Sets the current text.
        self.image_num_txt.setText('The Current Image is: ' + str(par_obj.curr_img +1))
        self.count_txt = QtGui.QLabel()
        
        
        self.output_count_txt = QtGui.QLabel()

        #Populates the grid with the different widgets.
        self.top_left_grid.addWidget(self.prev_im_btn, 0, 0)
        self.top_left_grid.addWidget(self.next_im_btn, 0, 1)
        self.top_left_grid.addWidget(self.image_num_txt, 2, 0, 2, 3)

        top_right_panel.setLayout(self.top_right_grid)

        self.eval_im_btn = QtGui.QPushButton('Evaluate Images')
        self.eval_im_btn.clicked.connect(self.evaluate_images)

        self.save_output_data_btn = QtGui.QPushButton('Save Output Data')
        self.save_output_data_btn.clicked.connect(self.save_output_data)

        self.save_output_link = QtGui.QLabel()
        self.save_output_link.setText('''<p><a href="'''+str(par_obj.csvPath)+'''">Goto output folder</a></p>
        <p><span style="font-size: 17px;"><br /></span></p>''')

        #Populates the grid on the right with the different widgets.
        self.top_right_grid.addWidget(self.eval_im_btn, 0, 0)
        self.top_right_grid.addWidget(self.save_output_data_btn, 1, 0)
        self.top_right_grid.addWidget(self.output_count_txt,2,0,1,4)
        #self.top_right_grid.addWidget(self.save_output_link, 2, 0)
        
        
        
        


        #Sets up the image panel splitter.
        image_panel = QtGui.QSplitter(QtCore.Qt.Horizontal)
        image_panel.addWidget(self.canvas1)
        image_panel.addWidget(self.canvas2)

        

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
        
    

    
        self.modelLoadedText = QtGui.QLabel(self)
        
        self.imageNumText = QtGui.QLabel(self)
        
        self.evalStatusText = QtGui.QLabel(self)
        
    def evaluate_images(self):
        par_obj.feat_arr = {}
        par_obj.pred_arr = {}
        par_obj.sum_pred = {}
        count = -1
        for b in par_obj.left_2_calc:
            frames =par_obj.frames_2_load[b]
            for i in frames:
                
                v2.im_pred_inline_fn(par_obj, self,inline=True,outer_loop=b,inner_loop=i,count=count)
                v2.evaluate_forest(par_obj,self, False, 0,inline=True,outer_loop=b,inner_loop=i,count=count)
                count = count+1
        v2.apply_correction(par_obj)

        self.save_output_data_btn.setEnabled(True)
        self.image_status_text.showMessage('Status: evaluation finished.')
        par_obj.eval_load_im_win_eval = True
        v2.eval_pred_show_fn(par_obj.curr_img, par_obj,self)
    def save_output_data(self):
        v2.save_output_data_fn(par_obj,self)
    def report_progress(self,message):
        self.image_status_text.showMessage('Status: ' + message)
        app.processEvents()
    def draw_saved_dots_and_roi(self):
        pass
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
        self.top_left_grid.addWidget(channel_wid,1,0,1,3)
    def prev_im_btn_fn(self):
        im_num = par_obj.curr_img - 1
        if im_num >-1:
            par_obj.curr_img = im_num
            v2.eval_goto_img_fn(im_num,par_obj,self)
            
    def next_im_btn_fn(self):
        im_num = par_obj.curr_img + 1
        if im_num <par_obj.test_im_end:
            par_obj.curr_img = im_num
            v2.eval_goto_img_fn(im_num,par_obj,self)

                  
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
            v2.eval_goto_img_fn(par_obj.curr_img,par_obj,evalImWin)
            
class File_Dialog(QtGui.QMainWindow):
    
    def __init__(self,par_obj,int_obj,modelTabObj):
        super(File_Dialog, self).__init__()
       
        
        self.int_obj = int_obj
        self.par_obj = par_obj
        self.type = 'im'
        self.modelTabObj = modelTabObj
        self.initUI()
        
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
        self.par_obj.config ={}
        try:
            self.par_obj.config = pickle.load(open(os.path.expanduser('~')+'/.densitycount/config.p', "rb" ));
            self.par_obj.filepath = self.par_obj.config['evalpath']
        except:
            self.par_obj.filepath = os.path.expanduser('~')+'/'
        #self.show()
        
    def showDialog(self):
        par_obj.file_array =[]
        self.int_obj.selIntButton.setEnabled(False)
        #filepath = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        for path in QtGui.QFileDialog.getOpenFileNames(self, 'Open file', self.par_obj.filepath,'Images(*.tif *.tiff *.png);;'):
            
            par_obj.file_array.append(path)
        
        self.par_obj.config['evalpath'] = str(QtCore.QFileInfo(path).absolutePath())+'/'
        pickle.dump(self.par_obj.config, open(str(os.path.expanduser('~')+'/.densitycount/config.p'), "w" ))
        self.par_obj.csvPath = self.par_obj.config['evalpath']




        v2.import_data_fn(par_obj,par_obj.file_array)
        par_obj.left_2_calc=[]
        par_obj.frames_2_load={}
        if par_obj.file_ext == 'png':
            for i in range(0,par_obj.file_array.__len__()):
                par_obj.left_2_calc.append(i)
                par_obj.frames_2_load[i] = [0]
            self.int_obj.image_status_text.showMessage('Status: Loading Images. Loading Image Num: '+str(par_obj.file_array.__len__()))
                
            
        elif par_obj.file_ext =='tiff' or par_obj.file_ext =='tif':
            if par_obj.tiff_file.maxFrames>1:
                for i in range(0,par_obj.file_array.__len__()):
                    par_obj.left_2_calc.append(i)
                    for b in range(0, par_obj.tiff_file.maxFrames):
                        par_obj.frames_2_load[i] = [0]
                    #try:
                    #    np.array(list(self.hyphen_range(fmStr)))-1
                    #    par_obj.frames_2_load[i] = np.array(list(self.hyphen_range(fmStr)))-1
                    #except:
                        #self.int_obj.image_status_text.showMessage('Status: The supplied range of image frames is in the wrong format. Please correct and click confirm images.')
                    #    return
                self.int_obj.image_status_text.showMessage('Status: Loading Images.')
               
               
            else:
                for i in range(0,par_obj.file_array.__len__()):
                    par_obj.left_2_calc.append(i)
                    par_obj.frames_2_load[i] = [0]
                
        count = 0
        for b in par_obj.left_2_calc:
            frames =par_obj.frames_2_load[b]
            for i in frames:
                count = count+1
                
        par_obj.test_im_start= 0
        par_obj.test_im_end= count
            
            
    
        self.int_obj.image_status_text.showMessage('Status: Loading Images. ')                
        if self.type == 'im':
            if(self.par_obj.file_array.__len__()>0):
                self.int_obj.selIntButton.setEnabled(True)
            

        self.refreshTable()
    def refreshTable(self):
        self.int_obj.image_status_text.showMessage(str(self.par_obj.file_array.__len__())+' Files Selected.')
        filesLen =self.par_obj.file_array.__len__()

        self.modelTabObj.show()
        
        c = 0
        
        for i in range(0,self.par_obj.file_array.__len__()):
                
                self.modelTabObj.setRowCount(c+1)
                btn = removeImBtn(self.int_obj,self.par_obj,self,i)
                btn.setText('Click to Remove')
                self.modelTabObj.setCellWidget(c, 0, btn)

                text1 = QtGui.QLabel(self.modelTabObj)
                text1.setText(str(self.par_obj.file_array[i]).split('/')[-1])
                self.modelTabObj.setCellWidget(c,1,text1)
                text2 = QtGui.QLabel(self.modelTabObj)
                text2.setText(str(self.par_obj.file_array[i]))
                self.modelTabObj.setCellWidget(c,2,text2)

                
                c= c+1
        self.modelTabObj.show()
        if self.type == 'im':
            if self.par_obj.file_array.__len__() ==0:
                self.modelTabObj.hide()               
                
                

class loadModelBtn(QtGui.QPushButton):
    def __init__(self,par_obj,int_obj,parent,idnum,fileName):
        QtGui.QPushButton.__init__(self,parent)
        self.par_obj = par_obj
        self.int_obj = int_obj
        self.modelNum = idnum
        self.fileName = fileName
        self.clicked.connect(self.onClick)
        self.type = []
    def onClick(self):
        self.int_obj.loadModelFn(self.par_obj,self.fileName)


class removeImBtn(QtGui.QPushButton):
    def __init__(self,parent,par_obj,table,idnum):
        QtGui.QPushButton.__init__(self,parent)
        self.modelNum = idnum
        self.par_obj = par_obj
        self.clicked.connect(self.onClick)
        self.table = table
    def onClick(self):
        self.par_obj.file_array.pop(self.modelNum)
        self.table.refreshTable()
 




class Parameter_class:
    def __init__(self):
        store = True
        
        
        self.evalLoadImWin_loaded = False
        self.evalLoadModelWin_loaded = False
        self.evalLoadImWin_loaded = False
        self.evalDispImWin_evaluated = False
        self.eval_load_im_win_eval = False
        self.y_limit = 1024
        self.x_limit = 1024
        self.featArr ={}
        self.pred_array={}
        self.sum_pred={}
        self.crop_x1 = 0
        self.crop_x2 = 0
        self.crop_y1 = 0
        self.crop_y2 = 0
        self.file_array =[]
        self.curr_img = 0
        
        self.left2calc = 0
        self.p_size = 1
        self.file_array =[]
        self.height = 0
        self.width = 0
        self.fresh_features = True
        self.forPath = os.path.expanduser('~')+'/.densitycount/models/'
        try:
            os.makedirs(self.forPath)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        self.csvPath = os.path.expanduser('~')+'/'
        



#Creates win, an instance of QWidget
class widgetSP(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.par_obj =[]
    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Period:
            im_num = self.par_obj.curr_img + 1
        if ev.key() == QtCore.Qt.Key_Comma:
            im_num = self.par_obj.curr_img - 1
        v2.evalGotoImgFn(im_num,self.par_obj, self)
    def wheelEvent(self, event):
        super(widgetSP, self).wheelEvent(event)
        if event.delta() < 0:
            im_num = self.par_obj.curr_img + 1
        if event.delta() > 0:
            im_num = self.par_obj.curr_img - 1
        v2.evalGotoImgFn(im_num,self.par_obj, self)



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

#Intialises counter object.
par_obj = Parameter_class()
#Main widgets.
evalLoadImWin = Eval_load_im_win(par_obj)
evalLoadModelWin = Eval_load_model_win(par_obj)
evalImWin = Eval_disp_im_win(par_obj)


#Adds win tab and places button in win.
win_tab.addTab(evalLoadImWin, "Select Images")
win_tab.addTab(evalLoadModelWin, "Load Model")
win_tab.addTab(evalImWin, "Evaluate Images")

#Defines size of the widget.
win_tab.resize(900, 600)

time.sleep(2.0)
splash.finish(win_tab)

#Initalises load screen.
#eval_load_im_win_fn(par_obj,evalLoadImWin)
#evalLoadModelWinFn(par_obj,evalLoadModelWin)
#evalDispImWinFn(par_obj,evalImWin)
win_tab.show()
# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
