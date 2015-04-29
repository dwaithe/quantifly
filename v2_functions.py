from PyQt4 import QtGui, QtCore, Qt
import PIL.Image
import numpy as np
import os
import vigra
import pylab
import csv
import time
from sklearn.ensemble import ExtraTreesRegressor
import cPickle as pickle
from scipy.ndimage import filters

"""QuantiFly Software v2.0

    Copyright (C) 2015  Dominic Waithe

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
def calculateCI(data1,data2, test_value):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    
    x = np.array(data1)
    y = np.array(data2)

    z = np.polyfit(x,y,1)
    p = np.poly1d(z)
    fit = p(x)

    # get the coordinates for the fit curve    
    c_y = [np.min(fit),np.max(fit)]
    c_x = [np.min(x),np.max(x)]

    # predict y values of origional data using the fit
    p_y = z[0] * x + z[1] 

    # calculate the y-error (residuals)
    y_err = y -p_y 

    #Take the input prediction, convert to error, so is in format of x.
    #p_xx = (p_x*z[0])+z[1]
    
    # now calculate confidence intervals for new test x-series
    mean_x = np.mean(x)         # mean of x
    n = len(x)              # number of samples in origional fit
    t = 2.31                # appropriate t value (where n=9, two tailed 95%)
    s_err = np.sum(np.power(y_err,2))   # sum of the squares of the residuals

    #This calculates the error for one point, pxx
    
    confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((( test_value)+mean_x),2)/ ((np.sum(np.power(x,2)))-n*(np.power(mean_x,2))))))
    
    #We then modulate the point we the calculated correction +/- the Confidence interval.
    p_up =  test_value +abs(confs)
    p_dn =  test_value -abs(confs)
    

    return p_up,p_dn

def apply_correction(par_obj,withGT=False):
    par_obj.CC ={}
    par_obj.CC1 ={}
    par_obj.CC2 ={}
    par_obj.lowerCI ={}
    par_obj.upperCI ={}
        
    for i in range(par_obj.test_im_start,par_obj.test_im_end):
        test_value = ((par_obj.sum_pred[i]*par_obj.M)+par_obj.c)
        p_dn,p_up  = calculateCI(par_obj.gt_vec, par_obj.error_vec, test_value);
    
        
        
        par_obj.CC[i] = par_obj.sum_pred[i]-((par_obj.sum_pred[i]*par_obj.M)+par_obj.c)
        par_obj.lowerCI[i] = abs(par_obj.sum_pred[i]-p_dn)-par_obj.CC[i]
        par_obj.upperCI[i] = abs(par_obj.sum_pred[i]-p_up)-par_obj.CC[i]
        
        print('corrected value: '+str(par_obj.CC[i]))
        if withGT == True:
            par_obj.CC_absErr = np.abs(par_obj.CC[i]-par_obj.gt_sum[i])
            par_obj.CC_perErr = (np.abs(par_obj.CC[i]-par_obj.gt_sum[i])*100)/par_obj.gt_sum[i]
            print('Ground Truth count: '+str(par_obj.gt_sum[i]))
            print('corrected Absolute ERROR: '+str(np.abs(par_obj.CC[i]-par_obj.gt_sum[i])))
            print('corrected Percentage ERROR: '+str((np.abs(par_obj.CC[i]-par_obj.gt_sum[i])*100)/par_obj.gt_sum[i]))

def make_correction(par_obj,model_num,withGT=False):
    #Makes linear correction to model data.
    par_obj.error_vec =[];
    pred_vec=[];
    par_obj.gt_vec =[];
   
    

    for b in range(0,par_obj.saved_ROI.__len__()):
        #Iterates through saved ROI.
        rects = par_obj.saved_ROI[b]
        par_obj.gt_vec.append(np.sum(par_obj.dense_array[rects[0]][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3]])/255)
        pred_vec.append(np.sum(par_obj.pred_arr[rects[0]][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3]])/255)
        par_obj.error_vec.append(pred_vec[-1]-par_obj.gt_vec[-1])
        
    #Fits straight-line to data.
    M, c = np.polyfit(par_obj.gt_vec, par_obj.error_vec,1)
    par_obj.M = M
    par_obj.c = c
    apply_correction(par_obj,withGT)


    
    


    
    


def save_roi_fn(par_obj):
    """Saves ROI"""

    #If there is no width or height either no roi is selected or it is too thin.
    if par_obj.rect_w != 0 and par_obj.rect_h != 0:
        #If the ROI was in the negative direction.
        if par_obj.rect_w < 0:
            s_ori_x = par_obj.ori_x_2
        else:
            s_ori_x = par_obj.ori_x
        if par_obj.rect_h < 0:
            s_ori_y = par_obj.ori_y_2
        else:
            s_ori_y = par_obj.ori_y

        #Finds the current frame and file.
        par_obj.rects = (par_obj.curr_img, int(s_ori_x), int(s_ori_y), int(abs(par_obj.rect_w)), int(abs(par_obj.rect_h)))
        return True
    
    return False
    

def update_training_samples_fn(par_obj,model_num):
    """Collects the pixels or patches which will be used for training and 
    trains the forest."""
    #Makes sure everything is refreshed for the training, encase any regions
    #were changed. May have to be rethinked for speed later on.
    par_obj.f_matrix =[]
    par_obj.o_patches=[]
    region_size = 0
    for b in range(0,par_obj.saved_ROI.__len__()):
        rects = par_obj.saved_ROI[b]
        region_size += rects[4]*rects[3]        
    
    calc_ratio = par_obj.limit_ratio_size
    
    #print 'calcratio',calc_ratio
    #print 'aftercratio',region_size/par_obj.limit_ratio_size

    for b in range(0,par_obj.saved_ROI.__len__()):

        #Iterates through saved ROI.
        rects = par_obj.saved_ROI[b]
        img2load = rects[0]



        #Loads necessary images only.
        try:
            par_obj.feat_arr[img2load]
        except:
            im_pred_inline_fn(par_obj,par_obj,True,img2load,0,img2load-1)

        if(par_obj.p_size == 1):
            #Finds and extracts the features and output density for the specific regions.
            mImRegion = par_obj.feat_arr[rects[0]][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3],:]
            denseRegion = par_obj.dense_array[rects[0]][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3]]
            #Find the linear form of the selected feature representation
            mimg_lin = np.reshape(mImRegion, (mImRegion.shape[0]*mImRegion.shape[1],mImRegion.shape[2]))
            #Find the linear form of the complementatory output region.
            dense_lin = np.reshape(denseRegion, (denseRegion.shape[0]*denseRegion.shape[1]))
            #Sample the input pixels sparsely or densely.
            if(par_obj.limit_sample == True):
                if(par_obj.limit_ratio == True):
                    par_obj.limit_size = round(mImRegion.shape[0]*mImRegion.shape[1]/calc_ratio,0)
                #Randomly sample from input ROI or im a certain number (par_obj.limit_size) patches. With replacement.
                indices =  np.random.choice(int(mImRegion.shape[0]*mImRegion.shape[1]), size=int(par_obj.limit_size), replace=True, p=None)
                #Add to feature vector and output vector.
                par_obj.f_matrix.extend(mimg_lin[indices])
                par_obj.o_patches.extend(dense_lin[indices])
            else:
                #Add these to the end of the feature Matrix, input patches
                par_obj.f_matrix.extend(mimg_lin)
                #And the the output matrix, output patches
                par_obj.o_patches.extend(dense_lin)
        if(par_obj.p_size >1):
            mgn = (win.p_size-1)/2
            #Finds the corresponding image.
            left_rect = rects[2]+1 -mgn
            right_rect = rects[2]+rects[4] +mgn+1
            top_rect = rects[1]+1 -mgn
            bot_rect = rects[1]+rects[3]+mgn+1
            if left_rect < 0:
                left_rect = 0
            if top_rect < 0:
                top_rect = 0
            if right_rect > par_obj.width - 1:
                right_rect = par_obj.width - 1
            if bot_rect > par_obj.height - 1:
                bot_rect = par_obj.height - 1
            mImRegion = win.par_obj.feat_arr[rects[0]][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3]]

            mImRegion = win.par_obj.feat_arr[rects[0]][left_rect:right_rect,top_rect:bot_rect,:]
            denseRegion = win.dense_array[rects[0]][left_rect:right_rect,top_rect:bot_rect]

            mimg_linPatch,dense_linPatch, pos = v2.extractPatch(win.p_size,mImRegion,denseRegion, 'sparse')
            
            win.f_matrix.extend(mimg_linPatch)
            win.o_patches.extend(dense_linPatch)
    #Sets up extra trees regressor object.
    
    par_obj.RF[model_num] = ExtraTreesRegressor(par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1)
    


    #Fits the data.
    t3 = time.time()
    print 'fmatrix',np.array(par_obj.f_matrix).shape
    print 'o_patches',np.array(par_obj.o_patches).shape
    par_obj.RF[model_num].fit(np.asfortranarray(par_obj.f_matrix), np.asfortranarray(par_obj.o_patches))
    t4 = time.time()
    print 'actual training',t4-t3 
def update_density_fn(par_obj):
   
    for im in par_obj.im_for_train:
        #Construct empty array for current image.
        dots_im = np.zeros((par_obj.height,par_obj.width))
        #In array of all saved dots.

        for i in range(0,par_obj.saved_dots.__len__()):
            #Any ROI in the present image.
            #print 'iiiii',win.saved_dots.__len__()
            if(par_obj.saved_ROI[i][0] == im):
                #Save the corresponding dots.
                dots = par_obj.saved_dots[i]
                #Scan through the dots
                for b in range(0,dots.__len__()):
                   
                    #save the column and row 
                    c_dot = dots[b][2]
                    r_dot = dots[b][1]
                    #Set it to register as dot.
                    dots_im[c_dot, r_dot] = 255
        #Convolve the dots to represent density estimation.
        dense_im = filters.gaussian_filter(dots_im.astype(np.float32),   float(par_obj.sigma_data), order=0, output=None, mode='reflect', cval=0.0)
        #Replace member of dense_array with new image.
        
        par_obj.dense_array[im] = dense_im

def im_pred_inline_fn(par_obj, int_obj,inline=False,outer_loop=None,inner_loop=None,count=None):
    """Accesses TIFF file slice or opens png. Calculates features to indices present in par_obj.left_2_calc"""
    if inline == False:
        outer_loop = par_obj.left_2_calc
        inner_loop_arr = par_obj.frames_2_load
        count = -1
    else:
        #par_obj.feat_arr ={}
        inner_loop_arr ={outer_loop:[inner_loop]}
        outer_loop = [outer_loop]


    #Goes through the list of files.
    for b in outer_loop:
            
            imStr = str(par_obj.file_array[b])
            frames = inner_loop_arr[b]
            for i in frames:
                count = count+1
                if par_obj.file_ext == 'tif' or par_obj.file_ext == 'tiff':
                    temp = Tiff_Controller(imStr)
                    imRGB = temp.get_frame(i)
                elif par_obj.file_ext == 'png':
                    imRGB = pylab.imread(str(imStr))*255
                if par_obj.fresh_features == False:
                    try:
                        #Try loading features.
                        time1 = time.time()
                        feat = pickle.load(open(imStr[:-4]+'_'+str(i)+'.p', "rb"))
                        time2 = time.time()
                        int_obj.report_progress('Loading Features for Image: '+str(b+1)+' Frame: ' +str(i+1))
            
                    except:
                        #If don't exist create them.
                        int_obj.report_progress('Calculating Features for Image: '+str(b+1)+' Frame: ' +str(i+1))
                        feat = feature_create(par_obj,imRGB,imStr,i)
                else:
                    #If you want to ignore previous features which have been saved.
                    int_obj.report_progress('Calculating Features for Image: '+str(b+1)+' Frame: ' +str(i+1))
                    feat =feature_create(par_obj,imRGB,imStr,i)
                par_obj.num_of_feat = feat.shape[2]
                par_obj.feat_arr[count] = feat  
    
    return
def feature_create(par_obj,imRGB,imStr,i):
    time1 = time.time()
    if par_obj.crop_x2 ==0 and par_obj.crop_x1 ==0:
            par_obj.crop_x1 = 0
            par_obj.crop_x2=imRGB.shape[1]
            par_obj.crop_y1 = 0
            par_obj.crop_y2=imRGB.shape[0]
    par_obj.height = par_obj.crop_y2-par_obj.crop_y1
    par_obj.width = par_obj.crop_x2-par_obj.crop_x1 
    
    if (par_obj.feature_type == 'basic'):
        feat = np.zeros(((int(par_obj.crop_y2)-int(par_obj.crop_y1)),(int(par_obj.crop_x2)-int(par_obj.crop_x1)),13*par_obj.ch_active.__len__()))
    if (par_obj.feature_type == 'fine'):
        feat = np.zeros(((int(par_obj.crop_y2)-int(par_obj.crop_y1)),(int(par_obj.crop_x2)-int(par_obj.crop_x1)),21*par_obj.ch_active.__len__()))
    if (par_obj.feature_type == 'fineSpatial'):
        feat = np.zeros(((int(par_obj.crop_y2)-int(par_obj.crop_y1)),(int(par_obj.crop_x2)-int(par_obj.crop_x1)),23*par_obj.ch_active.__len__()))
    
    for b in range(0,par_obj.ch_active.__len__()):
        if (par_obj.feature_type == 'basic'):
            imG = imRGB[:,:,par_obj.ch_active[b]].astype(np.float32)
            feat[:,:,(b*13):((b+1)*13)] = local_shape_features_basic(imG,par_obj.feature_scale)   
        if (par_obj.feature_type == 'fine'):
            imG = imRGB[:,:,par_obj.ch_active[b]].astype(np.float32)
            feat[:,:,(b*21):((b+1)*21)] = local_shape_features_fine(imG,par_obj.feature_scale)
        if (par_obj.feature_type == 'fineSpatial'):
            imG = imRGB[:,:,par_obj.ch_active[b]].astype(np.float32)
            feat[:,:,(b*23):((b+1)*23)] = local_shape_features_fine_spatial(imG,par_obj.feature_scale,i)
    #pickle.dump(feat,open(imStr[:-4]+'_'+str(i)+'.p', "wb"),protocol=2)
    return feat
    
def evaluate_forest(par_obj,int_obj,withGT,model_num,inline=False,inner_loop=None,outer_loop=None,count=None):
    if inline == False:
        outer_loop = par_obj.left_2_calc
        inner_loop_arr = par_obj.frames_2_load
        count = -1
    else:
        inner_loop_arr ={outer_loop:[inner_loop]}
        outer_loop = [outer_loop]

    #Finds the current frame and file.
    
    for b in outer_loop:
        frames =inner_loop_arr[b]
        for i in frames:
            count = count+1
            
            

            if(par_obj.p_size >1):
                
                mimg_lin,dense_linPatch, pos = extractPatch(par_obj.p_size, par_obj.feat_arr[count], None, 'dense')
                tree_pred = par_obj.RF[model_num].predict(mimg_lin)
                linPred = v2.regenerateImg(par_obj.p_size, tree_pred, pos)
                    
            else:
                mimg_lin = np.reshape(par_obj.feat_arr[count], (par_obj.height * par_obj.width, par_obj.feat_arr[count].shape[2]))
                t2 = time.time()
                linPred = par_obj.RF[model_num].predict(mimg_lin)
                t1 = time.time()
                


            par_obj.pred_arr[count] = linPred.reshape(par_obj.height, par_obj.width)

            maxPred = np.max(linPred)
            sum_pred =np.sum(linPred/255)
            par_obj.sum_pred[count] = sum_pred
            print 'prediction time taken',t1 - t2
            print 'Predicted count:',par_obj.sum_pred[count]
            int_obj.report_progress('Making Prediction for Image: '+str(b+1)+' Frame: ' +str(i+1))
                    

            if withGT == True:
                try:
                    #If it has already been opened.
                    a = par_obj.gt_sum[count]
                except:
                    #Else find the file.
                    gt_im =  pylab.imread(par_obj.gt_array[count])[:,:,0]
                    par_obj.gt_sum[count] = np.sum(gt_im)
                
                
                print('Ground Truth count: '+str(par_obj.gt_sum[count]))
                print('Absolute ERROR: '+str(np.abs(par_obj.sum_pred[count]-par_obj.gt_sum[count])))
                print('Percentage ERROR: '+str((np.abs(par_obj.sum_pred[count]-par_obj.gt_sum[count])*100)/par_obj.gt_sum[count]))
            

            

    
def regenerate_img(p_size,tree_pred,pos):
    outImg = np.zeros((evalImWin.par_obj.feat_arr[0].shape[0],evalImWin.par_obj.feat_arr[0].shape[1]))
    mgn = int((p_size-1)/2)
    norm = np.zeros((evalImWin.par_obj.feat_arr[0].shape[0],evalImWin.par_obj.feat_arr[0].shape[1]))
    zerot = np.ones((p_size,p_size))
    

    for i in range(0,tree_pred.shape[0]):
        y_pos = pos[i][0]
        x_pos = pos[i][1]
       
        outImg[y_pos-mgn:mgn+y_pos+1,x_pos-mgn:mgn+x_pos+1]  += np.array(tree_pred[i].reshape(p_size,p_size))
        norm[y_pos-mgn:mgn+y_pos+1,x_pos-mgn:mgn+x_pos+1]  += zerot
    

    
    ind2div = outImg > 0
    outImg[ind2div]= outImg[ind2div]*(1/norm[ind2div])
    
    return outImg


def extract_patch(p_size,mImRegion,denseRegion,sample):
    
    #patch margin.
    mgn = int((p_size-1)/2)
    
    #Active areas given patch margin.
    subImRegion =mImRegion[mgn:mImRegion.shape[0]-mgn-1,mgn:mImRegion.shape[1]-mgn-1,:]
    
    if denseRegion !=None:
        subDenseRegion = denseRegion[mgn:mImRegion.shape[0]-mgn-1,mgn:mImRegion.shape[1]-mgn-1]
    

    #Create meshgrid for quick index to position reference
    yR = np.arange(0,mImRegion.shape[0])
    xR = np.arange(0,mImRegion.shape[1])
    xvFull,yvFull = np.meshgrid(xR, yR)
    yv = yvFull[mgn:mImRegion.shape[0]-mgn-1,mgn:mImRegion.shape[1]-mgn-1]
    xv = xvFull[mgn:mImRegion.shape[0]-mgn-1,mgn:mImRegion.shape[1]-mgn-1]
    

    xvLin = xv.reshape(-1)
    yvLin = yv.reshape(-1)
    totalLocations = yv.shape[0]*yv.shape[1]
    
    if sample == 'sparse':
        #Samples non-densely.
        
        limit_size = np.floor(np.array((totalLocations/win.limit_ratio_size))).astype(np.int32)
        indices =  np.random.choice(totalLocations, size=limit_size, replace=True, p=None)
    elif(sample == 'dense'):
        indices = np.arange(0,totalLocations)

    #output containers
    mimgRegPatch =[]
    denseRegPatch = []
    pos =[]
    for i in range(0,indices.shape[0]):
        x_pos = xvLin[indices[i]]
        y_pos = yvLin[indices[i]]
        m_patch = mImRegion[y_pos-mgn:mgn+y_pos+1,x_pos-mgn:mgn+x_pos+1,:]
        if denseRegion !=None:
            dPatch = denseRegion[y_pos-mgn:mgn+y_pos+1,x_pos-mgn:mgn+x_pos+1]

        #if m_patch.shape[0]==2 and m_patch.shape[1]==2:
        mimgRegPatch.append(m_patch.reshape(-1))
        pos.append((y_pos,x_pos))
        if denseRegion !=None:
            denseRegPatch.append(dPatch.reshape(-1))
                
    
    return mimgRegPatch, denseRegPatch, pos


def local_shape_features_fine(im,scaleStart):
    #Exactly as in the Luca Fiaschi paper.
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,21))
    
    st08 = vigra.filters.structureTensorEigenvalues(im,s*1,s*2)
    st16 = vigra.filters.structureTensorEigenvalues(im,s*2,s*4)
    st32 = vigra.filters.structureTensorEigenvalues(im,s*4,s*8)
    st64 = vigra.filters.structureTensorEigenvalues(im,s*8,s*16)
    st128 = vigra.filters.structureTensorEigenvalues(im,s*16,s*32)
    
    f[:,:, 0]  = im
    f[:,:, 1]  = vigra.filters.gaussianGradientMagnitude(im, s)
    f[:,:, 2]  = st08[:,:,0]
    f[:,:, 3]  = st08[:,:,1]
    f[:,:, 4]  = vigra.filters.laplacianOfGaussian(im, s )
    f[:,:, 5]  = vigra.filters.gaussianGradientMagnitude(im, s*2) 
    f[:,:, 6]  =  st16[:,:,0]
    f[:,:, 7]  = st16[:,:,1]
    f[:,:, 8]  = vigra.filters.laplacianOfGaussian(im, s*2 )
    f[:,:, 9]  = vigra.filters.gaussianGradientMagnitude(im, s*4) 
    f[:,:, 10] =  st32[:,:,0]
    f[:,:, 11] =  st32[:,:,1]
    f[:,:, 12] = vigra.filters.laplacianOfGaussian(im, s*4 )
    f[:,:, 13]  = vigra.filters.gaussianGradientMagnitude(im, s*8) 
    f[:,:, 14] =  st64[:,:,0]
    f[:,:, 15] =  st64[:,:,1]
    f[:,:, 16] = vigra.filters.laplacianOfGaussian(im, s*8 )
    f[:,:, 17]  = vigra.filters.gaussianGradientMagnitude(im, s*16) 
    f[:,:, 18] =  st128[:,:,0]
    f[:,:, 19] =  st128[:,:,1]
    f[:,:, 20] = vigra.filters.laplacianOfGaussian(im, s*16 )
   
    
    
    return f
def local_shape_features_fine_spatial(im,scaleStart,im_num):
    #Exactly as in the Luca Fiaschi paper.
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,23))

    xv, yv = np.meshgrid(range(0,imSizeC), range(0,imSizeR))
    
    st08 = vigra.filters.structureTensorEigenvalues(im,s*1,s*2)
    st16 = vigra.filters.structureTensorEigenvalues(im,s*2,s*4)
    st32 = vigra.filters.structureTensorEigenvalues(im,s*4,s*8)
    st64 = vigra.filters.structureTensorEigenvalues(im,s*8,s*16)
    st128 = vigra.filters.structureTensorEigenvalues(im,s*16,s*32)
    
    f[:,:, 0]  = np.ones((imSizeC,imSizeR))*im_num
    f[:,:, 1]  = vigra.filters.gaussianGradientMagnitude(im, s)
    f[:,:, 2]  = st08[:,:,0]
    f[:,:, 3]  = st08[:,:,1]
    f[:,:, 4]  = vigra.filters.laplacianOfGaussian(im, s )
    f[:,:, 5]  = vigra.filters.gaussianGradientMagnitude(im, s*2) 
    f[:,:, 6]  =  st16[:,:,0]
    f[:,:, 7]  = st16[:,:,1]
    f[:,:, 8]  = vigra.filters.laplacianOfGaussian(im, s*2 )
    f[:,:, 9]  = vigra.filters.gaussianGradientMagnitude(im, s*4) 
    f[:,:, 10] =  st32[:,:,0]
    f[:,:, 11] =  st32[:,:,1]
    f[:,:, 12] = vigra.filters.laplacianOfGaussian(im, s*4 )
    f[:,:, 13] = vigra.filters.gaussianGradientMagnitude(im, s*8) 
    f[:,:, 14] =  st64[:,:,0]
    f[:,:, 15] =  st64[:,:,1]
    f[:,:, 16] = vigra.filters.laplacianOfGaussian(im, s*8 )
    f[:,:, 17] = vigra.filters.gaussianGradientMagnitude(im, s*16) 
    f[:,:, 18] =  st128[:,:,0]
    f[:,:, 19] =  st128[:,:,1]
    f[:,:, 20] = vigra.filters.laplacianOfGaussian(im, s*16 )
    f[:,:, 21] = xv
    f[:,:, 22] = yv

   
    
    
    return f
def local_shape_features_basic(im,scaleStart):
    #Exactly as in the Luca Fiaschi paper.
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,13))
    
    st08 = vigra.filters.structureTensorEigenvalues(im,s*1,s*2)
    st16 = vigra.filters.structureTensorEigenvalues(im,s*2,s*4)
    st32 = vigra.filters.structureTensorEigenvalues(im,s*4,s*8)
    
    
    f[:,:, 0]  = im
    f[:,:, 1]  = vigra.filters.gaussianGradientMagnitude(im, s)
    f[:,:, 2]  = st08[:,:,0]
    f[:,:, 3]  = st08[:,:,1]
    f[:,:, 4]  = vigra.filters.laplacianOfGaussian(im, s )
    f[:,:, 5]  = vigra.filters.gaussianGradientMagnitude(im, s*2) 
    f[:,:, 6]  =  st16[:,:,0]
    f[:,:, 7]  = st16[:,:,1]
    f[:,:, 8]  = vigra.filters.laplacianOfGaussian(im, s*2 )
    f[:,:, 9]  = vigra.filters.gaussianGradientMagnitude(im, s*4) 
    f[:,:, 10] =  st32[:,:,0]
    f[:,:, 11] =  st32[:,:,1]
    f[:,:, 12] = vigra.filters.laplacianOfGaussian(im, s*4 )
    
    
    
    return f

def eval_goto_img_fn(im_num, par_obj, int_obj):
    """Loads up and converts image to correct format"""

    #Finds the current frame and file.
    count = -1
    for b in par_obj.left_2_calc:
        frames =par_obj.frames_2_load[b]
        for i in frames:
            count = count+1
            if par_obj.curr_img == count:
                break;
        else:
            continue 
        break 
    

    
    if ( par_obj.file_ext == 'png'):
        imStr = str(par_obj.file_array[b])
        imRGB = pylab.imread(imStr)*255
    if ( par_obj.file_ext == 'tif' or  par_obj.file_ext == 'tiff'):
        imStr = str(par_obj.file_array[b])
        temp = Tiff_Controller(imStr)
        imRGB = temp.get_frame(i)
    count = 0
    CH = [0]*par_obj.numCH
    for c in range(0,par_obj.numCH):
        name = 'a = int_obj.CH_cbx'+str(c)+'.checkState()'
        exec(name)
        if a ==2:
            count = count + 1
            CH[c] = 1

    newImg =np.zeros((par_obj.height,par_obj.width,3))
    if count == 1:
        ch = par_obj.ch_active[0]
        if imRGB.shape> 2:
            newImg[:,:,0] = imRGB[:,:,ch]
            newImg[:,:,1] = imRGB[:,:,ch]
            newImg[:,:,2] = imRGB[:,:,ch]
        else:
            newImg[:,:,0] = imRGB
            newImg[:,:,1] = imRGB
            newImg[:,:,2] = imRGB
    else:
        if CH[0] == 1:
            newImg[:,:,0] = imRGB[:,:,0]
        if CH[1] == 1:
            newImg[:,:,1] = imRGB[:,:,1]
        if CH[2] == 1:
            newImg[:,:,2] = imRGB[:,:,2]

    
    par_obj.save_im = imRGB
    for i in range(0,int_obj.plt1.lines.__len__()):
        int_obj.plt1.lines.pop(0)
    par_obj.newImg = newImg
    int_obj.plt1.cla()
    int_obj.plt1.imshow(255-newImg)
    int_obj.draw_saved_dots_and_roi()
    int_obj.plt1.set_xticklabels([])
    int_obj.plt1.set_yticklabels([])
    int_obj.canvas1.draw()
    #del im
    
    
    int_obj.image_num_txt.setText('The Current image is No. ' + str(par_obj.curr_img+1)) # filename: ' +str(evalLoadImWin.file_array[im_num]))
    eval_pred_show_fn(im_num,par_obj,int_obj)

def eval_pred_show_fn(im_num,par_obj,int_obj):
    """Shows Prediction Image when forest is loaded"""
    if par_obj.eval_load_im_win_eval == True:
        int_obj.image_num_txt.setText('The Current Image is No. ' + str(par_obj.curr_img+1))
        
        string_2_show = 'The Predicted Count: ' + str(round(par_obj.sum_pred[im_num],1)) 
        if par_obj.upperCI[im_num] < 1000:
            string_2_show += ' with bias correction: '+str(round(par_obj.CC[im_num],1))+' +\- CI '+str(np.round(par_obj.upperCI[im_num],2))+''
        int_obj.output_count_txt.setText(string_2_show)
        int_obj.plt2.cla()
        int_obj.plt2.imshow(par_obj.pred_arr[im_num].astype(np.float32))
        int_obj.plt2.set_xticklabels([])
        int_obj.plt2.set_yticklabels([])
        int_obj.canvas2.draw()
        
 
def import_data_fn(par_obj,file_array):
    """Function which loads in Tiff stack or single png file to assess type."""
    prevExt = [] 
    prevBitDepth=[] 
    prevNumCH =[]
    for i in range(0,file_array.__len__()):
            n = str(i)
            imStr = str(file_array[i])
            par_obj.file_ext = imStr.split(".")[-1]
            
            if prevExt != [] and prevExt !=par_obj.file_ext:
                statusText = 'More than one file format present. Different number of image channels in the selected images'
                return False, statusText


            if par_obj.file_ext == 'tif' or par_obj.file_ext == 'tiff':
                par_obj.tiff_file = Tiff_Controller(imStr)
                par_obj.numCH = par_obj.tiff_file.im_sz[2]
                par_obj.bitDepth = par_obj.tiff_file.im.tag[0x102][0]

                if par_obj.tiff_file.im_sz[0] > par_obj.y_limit or par_obj.tiff_file.im_sz[1] > par_obj.x_limit:
                    statusText = 'Status: Your images are too large. Please reduce to less than 756x756.'
                    return False, statusText
                if par_obj.tiff_file.maxFrames > 8:
                    par_obj.uploadLimit = 8
                else:
                    par_obj.uploadLimit = par_obj.tiff_file.maxFrames


                 
                
                par_obj.test_im_end = par_obj.tiff_file.maxFrames
                imRGB = par_obj.tiff_file.get_frame(0)
                
                
            elif par_obj.file_ext =='png':
                 
                 imRGB = pylab.imread(imStr)*255
                 par_obj.test_im_end = file_array.__len__()
                 par_obj.numCH =imRGB.shape.__len__()
                 par_obj.bitDepth = 8

                 if imRGB.shape[0] > par_obj.y_limit or imRGB.shape[1] > par_obj.x_limit:
                    statusText = 'Status: Your images are too large. Please reduce to less than 756x756.'
                    return False, statusText
                
            else:
                 statusText = 'Status: Image format not-recognised. Please choose either png or TIFF files.'
                 return False, statusText

            #Error Checking File Extension
            par_obj.prevExt = par_obj.file_ext
            #Error Checking number of cahnnels.
            if prevNumCH != [] and prevNumCH !=par_obj.numCH:
                statusText = 'More than one file format present. Different number of image channels in the selected images'
                return False, statusText
            prevNumCH  = par_obj.numCH
            #Error Checking Bit Depth.
            if prevBitDepth != [] and prevBitDepth != par_obj.bitDepth:
                statusText = 'More than one file format present. Different bit-depth in these different images'
                return False, statusText
            prevBitDepth = par_obj.bitDepth
            
            
    #Creates empty array to record density estimation.
    
    par_obj.test_im_start = 0
    par_obj.height = imRGB.shape[0]
    par_obj.width = imRGB.shape[1]
    par_obj.im_num_range = range(par_obj.test_im_start, par_obj.test_im_end)
    par_obj.num_of_train_im = par_obj.test_im_end
    
    
    if imRGB.shape.__len__() > 2:
    #If images have more than three channels. 
        if imRGB.shape[2]>1:
            #If the shape of the third dimension is greater than 2.
            par_obj.ex_img = imRGB[:,:,:]
        else:
            #If the size of the third dimenion is just 1, this is invalid for imshow show we have to adapt.
            par_obj.ex_img = imRGB[:,:,0]
    

    
    statusText= str(file_array.__len__())+' Files Loaded.'
    return True, statusText

def save_output_data_fn(par_obj,int_obj):
    local_time = time.asctime( time.localtime(time.time()) )

    with open(par_obj.csvPath+'outputData.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([str(par_obj.selectedModel)]+[str('Filename: ')]+[str('Frame number: ')]+[str('Predicted count: ')]+[str('Corrected count: ')]+[str('CI ')])
    
    count = -1
    for b in par_obj.left_2_calc:
        frames =par_obj.frames_2_load[b]
        imStr = str(par_obj.file_array[count])
        for i in frames:
            count = count+1
            n = str(count)
            string = par_obj.csvPath+'output' + n.zfill(3)+'.tif'
            im_to_save= PIL.Image.fromarray(par_obj.pred_arr[count].astype(np.float32))
            im_to_save.save(string)
               
            with open(par_obj.csvPath+'outputData.csv', 'a') as csvfile:
                spamwriter = csv.writer(csvfile,  dialect='excel')
                spamwriter.writerow([local_time]+[str(imStr)]+['-'+str(i)]+[par_obj.sum_pred[count]]+[par_obj.CC[count]]+[par_obj.upperCI[count]])
                

    int_obj.report_progress('Data exported to '+ par_obj.csvPath)

        

    

class Tiff_Controller:
    def __init__(self,fname):
        '''fname is the full path '''
        self.im  = PIL.Image.open(fname)
        self.fname = fname
        self.im.seek(0)
        self.im_sz = [self.im.tag[0x101][0], self.im.tag[0x100][0],self.im.tag[0x102].__len__()]
        self.cur = self.im.tell()
        num = 0
        self.maxFrames =1
        while True:
            num = num+1
            try:
                self.im.seek(num)
            except EOFError:
                return None
            self.maxFrames = num
    def get_frame(self,j):
        '''Extracts the jth frame from the image sequence.
        if the frame does not exist return None'''
        try:
            self.im.seek(j)
        except EOFError:
            return None

        self.cur = self.im.tell()
        return np.reshape(self.im.getdata(),self.im_sz)
    def __iter__(self):
        self.im.seek(0)
        self.old = self.cur
        self.cur = self.im.tell()
        return self

    def next(self):
        try:
            self.im.seek(self.cur)
            self.cur = self.im.tell()+1
        except EOFError:
            self.im.seek(self.old)
            self.cur = self.im.tell()
            raise StopIteration
        return np.reshape(self.im.getdata(),self.im_sz)    
