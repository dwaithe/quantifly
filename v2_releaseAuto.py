import numpy as np
from PyQt4 import QtGui, QtCore
import time
import pylab
import random
import csv
import v2_functions as v2
import cPickle as pickle
import PIL


def make_correction(parObj,model_num):
    #Need full evaluation.
   

    error_vec =[];
    pred_vec=[];
    gt_vec =[];
    im_start = par_obj.test_im_start
    im_end = par_obj.test_im_end 
   
    for i in par_obj.to_process:
        par_obj.test_im_start = i
        par_obj.test_im_end = i+1

        v2.evaluate_forest(par_obj,par_obj,True,model_num);
    par_obj.test_im_start = im_start
    par_obj.test_im_end = im_end
    for ind in par_obj.sum_pred:
        gt_vec.append(par_obj.gt_sum[ind])
        pred_vec.append(par_obj.sum_pred[ind])
        error_vec.append(par_obj.sum_pred[ind]-par_obj.gt_sum[ind])
    
    M, c = np.polyfit(gt_vec, error_vec,1)
    M1, M2, c1, c2 = v2.bootstrap(gt_vec, error_vec, 100000, np.polyfit, 0.05)
    
    par_obj.CC = par_obj.PC-((par_obj.PC*M)+c)
    par_obj.CC1 = par_obj.PC-((par_obj.PC*M1)+c1)
    par_obj.CC2 = par_obj.PC-((par_obj.PC*M2)+c2)

    par_obj.CC_absErr = np.abs(par_obj.CC1-par_obj.gt_sum[im_start])
    par_obj.CC_perErr = (np.abs(par_obj.CC1-par_obj.gt_sum[im_start])*100)/par_obj.gt_sum[im_start]

    par_obj.CC1_absErr = np.abs(par_obj.CC1-par_obj.gt_sum[im_start])
    par_obj.CC1_perErr = (np.abs(par_obj.CC1-par_obj.gt_sum[im_start])*100)/par_obj.gt_sum[im_start]

    par_obj.CC2_absErr = np.abs(par_obj.CC2-par_obj.gt_sum[im_start])
    par_obj.CC2_perErr = (np.abs(par_obj.CC2-par_obj.gt_sum[im_start])*100)/par_obj.gt_sum[im_start]
    #par_obj.CC_absErr = np.abs(par_obj.CC-par_obj.gt_sum[im_start])
    #par_obj.CC_perErr = (np.abs(par_obj.CC-par_obj.gt_sum[im_start])*100)/par_obj.gt_sum[im_start]


    print('Ground Truth count: '+str(par_obj.gt_sum[im_start]))
    print('corrected Absolute ERROR: '+str(np.abs(par_obj.CC-par_obj.gt_sum[im_start])))
    print('corrected Percentage ERROR: '+str((np.abs(par_obj.CC-par_obj.gt_sum[im_start])*100)/par_obj.gt_sum[im_start]))
    

def save_dots_fn():
    """Saves dots in ROI"""
    par_obj.saved_dots.append(par_obj.dots)
    par_obj.saved_ROI.append(par_obj.rects)
    par_obj.draw_ROI = True
    par_obj.draw_dots = False
    par_obj.dots_past = par_obj.dots
    par_obj.dots = []
    par_obj.rects = np.zeros((1, 4))
    par_obj.ori_x = 0
    par_obj.ori_y = 0
    par_obj.rect_w = 0
    par_obj.rect_h = 0
    #Now we update a density image of the current Image.
    update_density_fn()
def update_density_fn():
    #Construct empty array for current image.
    par_obj.im_for_train = [par_obj.curr_img]
    v2.update_density_fn(par_obj)
    

    jim = PIL.Image.fromarray((par_obj.dense_array[par_obj.curr_img])*255)
    jim.save(par_obj.folder_str+par_obj.file_str_arr[par_obj.curr_img]+'output4.tiff')

def forestVarFn():

    #Shape of image, used a lot.
    #mimgHeight = par_obj.feat_arr[par_obj.curr_img].shape[0]
    #mimgWidth = par_obj.feat_arr[par_obj.curr_img].shape[1]
    #num_of_feat =  par_obj.feat_arr[par_obj.curr_img].shape[2]
    mimg_lin = np.reshape(par_obj.feat_arr[par_obj.curr_img], (par_obj.width * par_obj.height, par_obj.num_of_feat))
    #Initialise the linear array to collect prediction from the individual trees.
    tree_pred = np.zeros(( par_obj.height*par_obj.width, par_obj.num_of_tree))
    for c in range(0, par_obj.num_of_tree):
        
        tree_pred[:,c] = par_obj.RF[c].predict(mimg_lin)
        
    var_tree_lin = np.zeros((par_obj.height*par_obj.width))
    var_tree_lin = np.var(tree_pred, 1)
    #Reshape back to a 2D image.
    var_tree_im = var_tree_lin.reshape(par_obj.height, par_obj.width)
    #Write images to file.
    return var_tree_im

    



def rand_roi_fn():
    """Generates totally random ROI from the loaded data."""
    rand_width = random.randrange(par_obj.start_r, par_obj.end_r, 1)
    rand_height = random.randrange(par_obj.start_r, par_obj.end_r, 1)
    rand_x = random.randrange(0, par_obj.width-rand_width, 1)
    rand_y = random.randrange(0, par_obj.height-rand_height, 1)
    #We save this so we can redraw the rects later.
    par_obj.ori_x = rand_x
    par_obj.ori_y = rand_y
    par_obj.rect_h = rand_height
    par_obj.rect_w = rand_width
    par_obj.curr_img = par_obj.im_num_range[random.randrange(0, par_obj.im_num_range.__len__(), 1)]
    
def suggestROIFn():
    print('selecting ROI with Active Learning')
    
    area = []
    integral = []
    pixel_den = []

    #Calculate densities from existing ROI.
    for b in range(0, par_obj.saved_ROI.__len__()):
        #Iterates through saved ROI.
        rects = par_obj.saved_ROI[b]
        #This is convolving the maximas to get the density image.
        dense_im = par_obj.dense_array[rects[0]][rects[2]+1:rects[2]+rects[4], rects[1]+1:rects[1]+rects[3]]
        #Find the linear form of the selected feature representation
        dense_lin = np.reshape(dense_im, (-1, 1))

        area.append(dense_lin.shape[0])
        integral.append(sum(dense_lin)/255)
        pixel_den.append((integral[-1]/(area[-1])))
        print('the integral is:' +str(integral[-1]))
        print('the area is:' +str(area[-1]))
        print('This is the density/pixel:' +str(pixel_den[-1]))

    #Now we want to randomly sample from the training range.
    NUM_OF_EFFORTS = 300
    pred_area = []
    pred_integral = []
    pred_pixel_den = []
    pred_mag_diff = []
    save_rand_rect = []
    
    print('The image range: ' +str(par_obj.im_num_range))

    for d in range(0, NUM_OF_EFFORTS):
        #Samples randomly from the available range.
        im_num = par_obj.im_num_range[random.randrange(0, par_obj.im_num_range.__len__(), 1)]
        rand_width = random.randrange(par_obj.start_r, par_obj.end_r, 1)
        rand_height = random.randrange(par_obj.start_r, par_obj.end_r, 1)
        rand_x = random.randrange(0, par_obj.width - rand_width, 1)
        rand_y = random.randrange(0, par_obj.height - rand_height,1)
        rand_rect = [im_num, rand_x, rand_y, rand_width, rand_height]
        
        #Looks in predicted images to return region.
        pred_region = par_obj.pred[im_num][rand_y+1:rand_y+rand_height, rand_x+1:rand_x+rand_width]
        pred_area.append(pred_region.shape[0]*pred_region.shape[1])
        #Calculates the predicted integral
        pred_integral.append(sum(pred_region.reshape(-1, 1))/255)
        #Calculates the density.
        pred_pixel_den.append((pred_integral[-1]/(pred_area[-1])))
        save_rand_rect.append(rand_rect)
        np_pixel_den  = np.asarray(pixel_den)
        #Calculates the distance of the current region from 
        pred_mag_diff.append(sum(np.abs(np_pixel_den[:]-pred_pixel_den[-1])))
        #print('These are the suggested: density/pixel areas:'+ str(pred_pixel_den[-1]) )
        #print('These are the suggested: difference inMagnitude areas:'+ str(pred_mag_diff[-1]) )
      
    print('The best candidate is:'+str(np.argmax(pred_mag_diff)))
    print('The best candidate is:'+str(np.max(pred_mag_diff)))

    best_ind = np.argmax(pred_mag_diff)

    #We save this so we can redraw the rects later.
    par_obj.ori_x = save_rand_rect[best_ind][1]
    par_obj.ori_y = save_rand_rect[best_ind][2]
    par_obj.rect_w = save_rand_rect[best_ind][3]
    par_obj.rect_h = save_rand_rect[best_ind][4]
    par_obj.curr_img = save_rand_rect[best_ind][0]
    

def segActiveDenFn():
        
    col = [0, 128, 128]
    area = []
    integral = []
    pixel_den = []


    #Calculate densities from existing ROI.
    for b in range(0, par_obj.saved_ROI.__len__()):
        #Iterates through saved ROI.
        rects = par_obj.saved_ROI[b]
        #This is convolving the maximas to get the density image.
        dense_im =  par_obj.dense_array[rects[0]][rects[2]+1:rects[2]+rects[4], rects[1]+1:rects[1]+rects[3]]
        #Find the linear form of the selected feature representation
        dense_lin = np.reshape(dense_im, (-1, 1))

        area.append(dense_lin.shape[0])
        integral.append(sum(dense_lin)/255)
        pixel_den.append((integral[-1]/(area[-1])))
       

    #Now we want to randomly sample from the training range.
    #NUM_OF_EFFORTS = 300

    pred_area = []
    pred_integral = []
    pred_pixel_den = []
    pred_mag_diff = []
    save_rand_rect = []
    
    print('The image range: ' +str(par_obj.im_num_range))

    for d in range(0, par_obj.subdivide_ROI.__len__()):
        #Samples randomly from the available range.
        im_num = par_obj.subdivide_ROI[d][0]
        rect_width = par_obj.subdivide_ROI[d][3]
        rect_height = par_obj.subdivide_ROI[d][4]
        rect_x = par_obj.subdivide_ROI[d][1]
        rect_y = par_obj.subdivide_ROI[d][2]
        rand_rect = [im_num,rect_x,rect_y,rect_width,rect_height]
        
        #mImgRegion = par_obj.feat_arr[im_num][rand_y+1:rand_y+rand_height, rand_x+1:rand_x+rand_width,:]
        #mImgLin = np.reshape(mImgRegion, (mImgRegion.shape[0]*mImgRegion.shape[1],mImgRegion.shape[2]))
        
        #Looks in predictd images to return region.
        pred_region = par_obj.pred[im_num][rect_y+1:rect_y+rect_height, rect_x+1:rect_x+rect_width]
        
        
        pred_area.append(pred_region.shape[0]*pred_region.shape[1])
        #Calculates the predicted integral
        pred_integral.append(sum(pred_region.reshape(-1,1))/255)
        #Calculates the density.
        pred_pixel_den.append((pred_integral[-1]/(pred_area[-1])))
        save_rand_rect.append(rand_rect)

        np_pixel_den = np.asarray(pixel_den)
        
        #Calculates the distance of the current region from 
        pred_mag_diff.append(sum(np.abs(np_pixel_den[:]-pred_pixel_den[-1])))
        #print('These are the suggested: density/pixel areas:'+ str(pred_pixel_den[-1]) )
        #print('These are the suggested: difference inMagnitude areas:'+ str(pred_mag_diff[-1]) )
      
        
    print('The best candidate is:'+str(np.argmax(pred_mag_diff)))
    print('The best candidate is:'+str(np.max(pred_mag_diff)))

    best_ind = np.argmax(pred_mag_diff)
    par_obj.subdivide_ROI.pop(best_ind)
    #We save this so we can redraw the rects later.
    par_obj.ori_x = save_rand_rect[best_ind][1]
    par_obj.ori_y = save_rand_rect[best_ind][2]
    par_obj.rect_w = save_rand_rect[best_ind][3]
    par_obj.rect_h = save_rand_rect[best_ind][4]
    par_obj.curr_img = save_rand_rect[best_ind][0]
    

    
def seg_active_var_fn():
    
    print('Choosing image segments using Variance Active Learning')

    col = [0, 128, 128]
    area = []
    integral = []
    pixel_den = []


    #Calculate densities from existing ROI.
    for b in range(0, par_obj.saved_ROI.__len__()):
        #Iterates through saved ROI.
        rects = par_obj.saved_ROI[b]
        #This is convolving the maximas to get the density image.
        dense_im =  par_obj.dense_array[rects[0]][rects[2]+1:rects[2]+rects[4], rects[1]+1:rects[1]+rects[3]]
        #Find the linear form of the selected feature representation
        dense_lin = np.reshape(dense_im, (-1, 1))

        area.append(dense_lin.shape[0])
        integral.append(sum(dense_lin)/255)
        pixel_den.append((integral[-1]/(area[-1])))
        print('the integral is:' +str(integral[-1]))
        print('the area is:' +str(area[-1]))
        print('This is the density/pixel:' +str(pixel_den[-1]))

    #Now we want to randomly sample from the training range.
    #NUM_OF_EFFORTS = 300

    pred_area = []
    pred_integral = []
    pred_pixel_den = []
    pred_mag_diff = []
    save_rand_rect = []
    
    print('The image range: ' +str(par_obj.im_num_range))

    for d in range(0, par_obj.subdivide_ROI.__len__()):
        #Samples randomly from the available range.
        im_num = par_obj.subdivide_ROI[d][0]
        rect_width = par_obj.subdivide_ROI[d][3]
        rect_height = par_obj.subdivide_ROI[d][4]
        rect_x = par_obj.subdivide_ROI[d][1]
        rect_y = par_obj.subdivide_ROI[d][2]
        rand_rect = [im_num, rect_x, rect_y, rect_width, rect_height]    
        #Looks in predictd images to return region.
        var_region = par_obj.var[im_num][rect_y+1:rect_y+rect_height, rect_x+1:rect_x+rect_width]
        pred_area.append(var_region.shape[0]*var_region.shape[1])
        #Calculates the predicted integral
        pred_integral.append(sum(var_region.reshape(-1, 1)))
        #Calculates the Variance/area.
        pred_pixel_den.append((pred_integral[-1]/(pred_area[-1])))
        save_rand_rect.append(rand_rect)

        
        
    print('The best candidate is:'+str(np.argmax(pred_pixel_den)))
    print('The best candidate is:'+str(np.max(pred_pixel_den)))

    best_ind = np.argmax(pred_pixel_den)
    #We remove the winning rectangle
    par_obj.subdivide_ROI.pop(best_ind)

    #We save this so we can redraw the rects later.
    par_obj.ori_x = save_rand_rect[best_ind][1]
    par_obj.ori_y = save_rand_rect[best_ind][2]
    par_obj.rect_w = save_rand_rect[best_ind][3]
    par_obj.rect_h = save_rand_rect[best_ind][4]
    par_obj.curr_img = save_rand_rect[best_ind][0]
    

    
def im_ROI(im_num):
    
    print('Whole image region.')
    par_obj.ori_x = 0
    par_obj.ori_y = 0
    par_obj.rect_w = par_obj.width
    par_obj.rect_h = par_obj.height
    par_obj.curr_img = im_num
    
    
    
   
def seg_rand_im_gen_fn(im_num):
    #col_start = 0
    #row_start = 0
    #col_end = par_obj.height
    #row_end = par_obj.width
    #subdivide(col_start, col_end, row_start, row_end, im_num)
    ori_x = 0
    ori_y = 0
    rect_h = par_obj.height
    rect_w = par_obj.width

    par_obj.subdivide_ROI.append([ori_x, ori_y, np.round(rect_w/2,0),  np.round(rect_h/2,0), im_num])
    par_obj.subdivide_ROI.append([np.round(rect_w/2,0), ori_y,  np.round(rect_w/2,0), np.round(rect_h/2,0), im_num])
    par_obj.subdivide_ROI.append([np.round(rect_w/2,0), np.round(rect_h/2,0),  np.round(rect_w/2,0), np.round(rect_h/2,0), im_num])
    par_obj.subdivide_ROI.append([ori_x, ori_y,  np.round(rect_w/2,0), np.round(rect_h/2,0), im_num])

def subdivide(col_start, col_end, row_start, row_end, im_num):
    #Recursive function.
    col_divide = np.random.randint(col_start+(par_obj.roi_min_size)/2, col_end-(par_obj.roi_min_size)/2, size=1)
    row_divide = np.random.randint(row_start+(par_obj.roi_min_size)/2, row_end-(par_obj.roi_min_size)/2, size=1)
    #Top left
    ori_x = row_start
    ori_y = col_start
    rect_w = row_divide - row_start
    rect_h = col_divide - col_start
    if(np.random.rand(1)>par_obj.subdivide_prob and rect_w>par_obj.roi_min_size and rect_h>par_obj.roi_min_size):
        subdivide(ori_y, (ori_y +rect_h), ori_x, (ori_x+rect_w), im_num)
    else:
        par_obj.subdivide_ROI.append([ori_x, ori_y, rect_w, rect_h, im_num])
    #Top Right
    ori_x = row_divide
    ori_y = col_start
    rect_w = row_end - row_divide
    rect_h = col_divide - col_start
    if(np.random.rand(1)>par_obj.subdivide_prob and rect_w>par_obj.roi_min_size and rect_h>par_obj.roi_min_size ):
        subdivide(ori_y, (ori_y +rect_h), ori_x, (ori_x+rect_w), im_num)
    else:
        par_obj.subdivide_ROI.append([ori_x, ori_y, rect_w, rect_h, im_num])
    #Bottom Left
    ori_x = row_start
    ori_y = col_divide 
    rect_w = row_divide - row_start
    rect_h = col_end - col_divide
    if(np.random.rand(1)>par_obj.subdivide_prob and rect_w>par_obj.roi_min_size and rect_h>par_obj.roi_min_size):
        subdivide(ori_y,(ori_y +rect_h), ori_x, (ori_x+rect_w), im_num)
    else:
        par_obj.subdivide_ROI.append([ori_x, ori_y, rect_w, rect_h, im_num])
    #Bottom Right
    ori_x = row_divide
    ori_y = col_divide
    rect_w = row_end - row_divide
    rect_h = col_end - col_divide
    if(np.random.rand(1)>par_obj.subdivide_prob and rect_w>par_obj.roi_min_size and rect_h>par_obj.roi_min_size ):
        subdivide(ori_y, (ori_y +rect_h), ori_x, (ori_x+rect_w), im_num)
    else:
        par_obj.subdivide_ROI.append([ori_x, ori_y, rect_w, rect_h, im_num])

def seg_rand_im_fn():
    if(par_obj.first_time == True):
        for im_num in par_obj.im_num_range:
                seg_rand_im_gen_fn(im_num)
        par_obj.first_time = False

    #sub_rect_ind = np.random.randint(0, par_obj.subdivide_ROI.__len__())
    #rects = par_obj.subdivide_ROI.pop(sub_rect_ind)
    rects = par_obj.subdivide_ROI.pop()
    par_obj.ori_x = int(rects[0])
    par_obj.ori_y = int(rects[1])
    par_obj.rect_w = int(rects[2])
    par_obj.rect_h = int(rects[3])
    par_obj.curr_img = int(rects[4])
    print par_obj.ori_x, par_obj.ori_y, par_obj.rect_w, par_obj.rect_h, par_obj.curr_img

def find_patch(image, i_pos, j_pos, mgn):
    im_reg_patch = []
    for b in range(0, i_pos.__len__()):
        m_patch = image[(j_pos[b]-mgn):mgn+j_pos[b]+1, (i_pos[b]-mgn):mgn+i_pos[b]+1]
        if m_patch.shape[0] == (mgn*2+1) and m_patch.shape[1] == (mgn*2+1):
            m_patch = m_patch.astype(np.float32)#/np.max(m_patch.reshape(-1))
            im_reg_patch.append(m_patch)
    return im_reg_patch

def calculate_quality(peak_find, par_obj):
    """Whats the quality of performance"""
    if peak_find == True:
        #Find candidate peaks before fitting the gaussian. Used when training data is not present.
        print 'needs writing'
    else:
    #Finds the patches from coordinates.
        i_pos = []
        j_pos = []
        im_reg_patch = []
        for key in par_obj.pred_arr:
            for i in range(0, par_obj.saved_dots.__len__()):
            #Any ROI in the present image.
                if(par_obj.saved_ROI[i][0] == key):
                #Save the corresponding dots.
                    dots = par_obj.saved_dots[i]
                    #Scan through the dots
                    for b in range(0, dots.__len__()):
                        #save the column and row 
                        j_pos.append(int(dots[b][2]))
                        i_pos.append(int(dots[b][1]))
                        #Set it to register as dot.
            #im_reg_patch.append(find_patch(pylab.imread(par_obj.gt_array[key])[:,:]*255, i_pos,j_pos,5))
            im_reg_patch.append(find_patch(par_obj.pred_arr[key], i_pos, j_pos, 5))
        pickle.dump(im_reg_patch, open( "save.p", "wb" ) )

def auto_fn():

    #Number of ROI to select.
    par_obj.total_count_dots =0
    for i in range(0, par_obj.num_of_iterations):

        par_obj.curr_iteration = i
        print 'curr_iteration', i
        
        if(par_obj.TYPE_ROI == 'segRandIm'):
            print 'segRandIm'
            seg_rand_im_fn()
        elif(par_obj.TYPE_ROI == 'wholeImROI'):
            im_ROI(par_obj.im_num_range[i])
            par_obj.to_process.append(par_obj.im_num_range[i])
        elif(par_obj.TYPE_ROI == 'randomROI'): 
            rand_roi_fn()
            
        elif(par_obj.TYPE_ROI == 'activeDenROI'):
            if(par_obj.curr_iteration==0):
                rand_roi_fn()
            else:
                suggestROIFn()
        elif(par_obj.TYPE_ROI == 'segActiveDen'):
            if(par_obj.first_time == True):
               
                seg_rand_im_fn()
            else:
                
                segActiveDenFn()
        elif(par_obj.TYPE_ROI == 'segActiveVar'):
            if(par_obj.first_time == True):
                print('here at segActiveVar')
               
                seg_rand_im_fn()
            else:
                
                seg_active_var_fn()

        success = v2.save_roi_fn(par_obj)
        #Read from dots images and apply instances.
        par_obj.dots = []
        par_obj.pred_arr = {}
        par_obj.sum_pred = {}
        par_obj.gt_sum = {}
        gt_im = pylab.imread(par_obj.gt_array[par_obj.curr_img])
        if gt_im.shape.__len__()>2:
            gt_im = gt_im[:, :, 0]
        par_obj.gt_sum[par_obj.curr_img] = np.sum(gt_im)

        
        
        col_start = par_obj.ori_y+1 - par_obj.roi_tolerance
        col_end = par_obj.ori_y+par_obj.rect_h + par_obj.roi_tolerance
        row_start = par_obj.ori_x+1 - par_obj.roi_tolerance
        row_end = par_obj.ori_x+par_obj.rect_w  + par_obj.roi_tolerance
        if(col_start < 0):
            col_start = 0
        if(col_end > par_obj.height):
            col_end = par_obj.height
        if(row_start < 0):
            row_start = 0
        if(row_end > par_obj.width):
            row_end = par_obj.width
        

        vec = np.where(gt_im[col_start:col_end, row_start:row_end] >0)

        for b in range(0, vec[0].__len__()):

            
            par_obj.dots.append([par_obj.curr_img, row_start+vec[1][b], col_start+vec[0][b]])
                    #Stores dots as image, over-writing duplicates.
            par_obj.dots_array[par_obj.curr_img][col_start+vec[0][b], row_start+vec[1][b]] = 1

            par_obj.total_count_dots +=1
            
        
        #Will save these dots and any previous ones.            
        save_dots_fn()
       
        if(par_obj.train_at_each_itr== True):
            par_obj.RF ={}
            total_error ={}
            for b in range(0,1):
                t1= time.time()
                model_num= b
                print 'what'
                v2.update_training_samples_fn(par_obj,model_num)
                t2 = time.time()
                print 'train forest: ',t2-t1
                v2.evaluate_forest(par_obj,par_obj, True,model_num,0)
                if par_obj.inc_bias_corr == True:
                        
                       
                        abs_error = 0
                        for l in range(par_obj.test_im_start, par_obj.test_im_end):
                            abs_error += np.abs(par_obj.sum_pred[l]-par_obj.gt_sum[l])
                        total_error[b] = abs_error

            print 'total error',total_error
            best_model = np.argmin(total_error)
            par_obj.PC = par_obj.sum_pred[par_obj.test_im_start]
            make_correction(par_obj,best_model)

            output_results()
           
        else:
            if(i in par_obj.iter_to_train_forest):
                print 'go on.'
                par_obj.RF ={}
                total_error ={}
                for b in range(0,1):
                    t1= time.time()
                    model_num= b
                    v2.update_training_samples_fn(par_obj,model_num)
                    t2 = time.time()
                                    
                    v2.evaluate_forest(par_obj,par_obj, True,model_num)
                v2.im_pred_inline_fn(par_obj, par_obj,inline=True, inner_loop=0, outer_loop=par_obj.test_im_start,count=par_obj.test_im_start-1)
                v2.evaluate_forest(par_obj,par_obj, True,model_num, inline=True, inner_loop=0, outer_loop=par_obj.test_im_start,count=par_obj.test_im_start-1)
                    
                if par_obj.inc_bias_corr == True:
                    v2.make_correction(par_obj,model_num,True)
                    
                    output_results()

                #calculate_quality(False,par_obj)
                #break
        if(par_obj.TYPE_ROI == 'segActiveDen' and par_obj.subdivide_ROI == []):
            break
        if(par_obj.TYPE_ROI == 'segRandIm' and par_obj.subdivide_ROI == []):
            break
        if(par_obj.TYPE_ROI == 'segActiveVar' and par_obj.subdivide_ROI == []):
            break



class parameterClass:
    def __init__(self):
        #Sets parameters for automatic experiment.'wholeImROI', 'activeDenROI', 'randomROI', 'segRandIm'.'segActiveDen','segActiveVar'
        self.TYPE_ROI = 'wholeImROI'
        #ROI parameters, defines minimum and max size if set to random or active Learning. Tolerance is border region size
        self.start_r = 5
        self.end_r = 200
        self.roi_tolerance = 10
        self.roi_min_size = 80 #40
        self.subdivide_prob = 0.0
        #Parameters of sampling
        self.limit_sample = True
        self.limit_ratio = True #whether to use ratio of roi pixels
        self.limit_ratio_size = 21#Gives 3000 patches for 255*255 image.
        self.limit_size = 3000 #patches per image or ROI.
        #Random Forest parameters
        self.pw = 1
        self.max_depth = 10
        self.min_samples_split = 20 
        self.min_samples_leaf = 10  
        self.max_features = 7
        #Allows a seed to be defined giving reproducibility in the random forest. 
        self.num_of_tree = 30
        #If leave one Out is true, it will train with all images at test on one.
        self.cross_valid_method = 'leaveOneOut'
        self.replace_im = False
        self.csvPath = '/Users/dwaithe/Documents/collaborators/PiperM/analysis/output'
        self.csv_file_name = 'DOM3_ImageComp.csv'
        
        #File location
        self.folder_str = None
        self.feature_type = None
        self.sigma_data = None
        self.feature_scale = None
        self.folder_str = None
        self.num_of_im = None
        self.test_im_start = None
        self.test_im_end = None
        self.file_ext = None
        self.x_limit = None
        self.y_limit = None
        self.num_of_experiments = None
        self.num_of_iterations = None
        self.num_of_train_im = None
        self.curr_iteration = None
        self.train_at_each_itr = None
        #Auto mode.
        self.auto = True
        self.height = 0
        self.width = 0
        self.crop_x1 = 0
        self.crop_y1 = 0
        self.crop_x2 = 0
        self.crop_y2 = 0
        self.p_size = 1
        self.ori_x = None
        self.ori_y = None
        self.ori_x_2 = None
        self.ori_y_2 = None
        self.rect_w = None
        self.rect_h = None
        self.curr_img = None
        self.curr_experiment = None
        self.inc_bias_corr = False
        self.fresh_features = True


        self.gt_array = []
        self.gt_sum = []
        self.file_array = []
        self.ch_active = []
        self.file_str_arr = []
        self.dots_array = []
        self.dots_count_arr = []
        self.dense_array = []
        self.sum_pred = []
        self.dots_past = []
        self.to_process =[]
        self.im_for_train = []

        self.feat_arr = {}
    def report_progress(self,message):
        print message

def output_header():
    """Prints header for each iteration and output for each data-point."""
    local_time = time.asctime( time.localtime(time.time()) )
    spamwriter = open(par_obj.csvPath+par_obj.csv_file_name, 'a')
    spamwriter.write(str('time')+','+str('file name:')+','+str('ground-truth')+','+str('prediction')+','+str('abs_error')+','+str('per_error')+','+str('numOfDots')+','+str('corrCount')+','+str('lowerCI')+','+str('upperCI')+'\n')
    spamwriter.close()

def output_results():
    
    
    for l in range(par_obj.test_im_start, par_obj.test_im_end):
        print 'reported result',l
        abs_error = np.abs(par_obj.sum_pred[l]-par_obj.gt_sum[l])
        per_error = (np.abs(par_obj.sum_pred[l]-par_obj.gt_sum[l])*100)/par_obj.gt_sum[l]
        local_time = time.asctime(time.localtime(time.time()))
        jim = PIL.Image.fromarray((par_obj.pred_arr[l])*255)
        jim.save(par_obj.folder_str+par_obj.file_str_arr[l]+'output.tiff')
        
        spamwriter = open(par_obj.csvPath+par_obj.csv_file_name, 'a')
        spamwriter.write(str(local_time)+','+str(par_obj.file_str_arr[l])+','+str(par_obj.gt_sum[l])+','+str(par_obj.sum_pred[l])+','+str(abs_error)+','+str(per_error)+','+str(par_obj.total_count_dots)+','+str(par_obj.CC[l])+','+str(par_obj.lowerCI[l])+','+str(par_obj.upperCI[l])+'\n')
        spamwriter.close()
        

def save_parameters():
    """Prints the header of the experiment"""

    save_parameters_line_1 = 'TYPE_ROI: '+str(par_obj.TYPE_ROI)+', par_obj.roi_min_size'+str(par_obj.roi_min_size)+', par_obj.start_r: '+str(par_obj.start_r)+', par_obj.end_r: '+str(par_obj.end_r)+', par_obj.roi_tolerance: '+str(par_obj.roi_tolerance)+', par_obj.subdivide_prob: '+str(par_obj.subdivide_prob)+' sigma_data: '+str(par_obj.sigma_data)+ ', feature Scale: '+str(par_obj.feature_scale)+', pw: ' +str(par_obj.pw)+', num_of_tree: '+str(par_obj.num_of_tree)+', max_depth: '+str(par_obj.max_depth)+', min_samples_leaf: '+str(par_obj.min_samples_leaf)+', min_samples_split: '+str(par_obj.min_samples_split)+', win_max_features: '+str(par_obj.max_features)
    save_parameters_line_2 = 'num_of_train_im: '+str(par_obj.num_of_train_im)+', num_of_iterations: '+str(par_obj.num_of_iterations)+', num_of_experiments: '+ str(par_obj.num_of_experiments)+', test_im_start: '+str(par_obj.test_im_start) +', test_im_end: '+str(par_obj.test_im_end)+', test_im_start: ' +str(par_obj.test_im_start) +', test_im_end: '+str(par_obj.test_im_end)+', replace image: '+str(par_obj.replace_im)+', folder_str: ' + str(par_obj.folder_str)
    save_parameters_line_3 = 'par_obj.cross_valid_method:  '+str(par_obj.cross_valid_method)+', par_obj.train_at_each_itr'+str(par_obj.train_at_each_itr)+', iter_to_train_forest: '+str(par_obj.iter_to_train_forest)+', feature_type: '+str(par_obj.feature_type)

    #Initialise experiment save files.
    local_time = time.asctime( time.localtime(time.time()) )
    with open(par_obj.csvPath+par_obj.csv_file_name, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_ALL)
            spamwriter.writerow([local_time]+[save_parameters_line_1])
            spamwriter.writerow([local_time]+[save_parameters_line_2])
            spamwriter.writerow([local_time]+[save_parameters_line_3])


par_obj = parameterClass()

for x in range(1, 2):
   # if(x==1):
    #    par_obj.sigma_data = 0.4
    #if(x==2):
     #   par_obj.sigma_data = 0.6
    #if(x==3):
     #   par_obj.sigma_data = 0.8
    #if(x==4):
     #   par_obj.sigma_data = 1.0
    #if(x==5):
     #   par_obj.sigma_data = 1.2
    
    for o in range(1, 2):
       # if (o==1):
         #   par_obj.feature_scale =0.4
        #if (o==2):
          #  par_obj.feature_scale =0.6
        #if (o==3):
         #   par_obj.feature_scale =0.8
        #if (o==4):
            #par_obj.feature_scale =1.0
        #if (o==5):
         #   par_obj.feature_scale =1.2
        for j in range(10,11):
            par_obj.gt_array = []
            par_obj.file_array = []
            par_obj.x_limit = 1024
            par_obj.y_limit = 1024
            par_obj.crop_x1 = 0
            par_obj.crop_y1 = 0
            par_obj.crop_x2 = 0
            par_obj.crop_y2 = 0
            par_obj.p_size = 1
            par_obj.feature_type = 'fine'
            
            if(j == 1):
                #dataset 1 DM Media.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../final_data/data01-20130531-DM/'
                par_obj.csvPath = '../analysis/'
                #par_obj.csv_file_name = 'imagesRequired-data1.csv'
                par_obj.csv_file_name = 'standard-data1.csv'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.file_str_arr = []
                par_obj.inc_bias_corr = True
                par_obj.train_at_each_itr = False
                par_obj.left_2_calc = np.arange(0,par_obj.num_of_im)
                par_obj.frames_2_load = [[0],[0],[0],[0],[0],[0],[0],[0]]

                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    par_obj.file_str_arr.append('out'+n.zfill(3))
                    par_obj.gt_array.append(par_obj.folder_str+n.zfill(3)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+n.zfill(3)+'cell.png')
                
            
            if j == 2:
                #dataset 2 DM Media.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/final_data/data02-20130709-DM/'
                par_obj.csvPath = '/Users/dwaithe/Documents/collaborators/PiperM/analysis/'
                #par_obj.csv_file_name = 'imagesRequired-data2.csv'
                par_obj.csv_file_name = 'standard-data2.csv'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.file_str_arr = []
                par_obj.inc_bias_corr = True
                par_obj.train_at_each_itr = False
                par_obj.left_2_calc = np.arange(0,par_obj.num_of_im)
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    par_obj.file_str_arr.append('out'+n.zfill(3))
                    par_obj.gt_array.append(par_obj.folder_str+n.zfill(3)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+n.zfill(3)+'cell.png')
            
            if(j == 3):
                #dataset 9 SY Media.
                #Original dataset without challenging artefacts
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/final_data/data03-20140331-DM/'
                par_obj.csvPath = '/Users/dwaithe/Documents/collaborators/PiperM/analysis/'
                #par_obj.csv_file_name = 'imagesRequired-data9.csv'
                par_obj.csv_file_name = 'standard-data3.csv'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.file_str_arr = []
                par_obj.inc_bias_corr = True
                par_obj.train_at_each_itr = False
                par_obj.left_2_calc = np.arange(0,par_obj.num_of_im)
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)

                    par_obj.file_str_arr.append('out'+n.zfill(3))
                    par_obj.gt_array.append(par_obj.folder_str+n.zfill(3)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+n.zfill(3)+'cell.png')

                #par_obj.iter_to_train_forest = [13]
            if(j == 4):
                #dataset 4 DM Media.
                #Original dataset without challenging artefacts
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/final_data/data04-20140331-DM/'
                par_obj.csvPath = '/Users/dwaithe/Documents/collaborators/PiperM/analysis/'
                #par_obj.csv_file_name = 'imagesRequired-data4.csv'
                par_obj.csv_file_name = 'standard-data4.csv'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.file_str_arr = []
                par_obj.inc_bias_corr = True
                par_obj.train_at_each_itr = False
                par_obj.left_2_calc = np.arange(0,par_obj.num_of_im)
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    par_obj.file_str_arr.append('out'+n.zfill(3))
                    par_obj.gt_array.append(par_obj.folder_str+n.zfill(3)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+n.zfill(3)+'cell.png')
            if j == 5:
                #dataset 5 DM Media.
                #purposefully selected images to express a large a range as possible.
                #For exemplifying non-artefact bias.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/final_data/data05-bias-DM/'
                par_obj.csvPath = '/Users/dwaithe/Documents/collaborators/PiperM/analysis/'
                #par_obj.csv_file_name = 'imagesRequired-data5.csv'
                par_obj.csv_file_name = 'standard-data5.csv'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.inc_bias_corr = True
                par_obj.train_at_each_itr = False
                par_obj.left_2_calc = np.arange(0,par_obj.num_of_im)
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    par_obj.file_str_arr.append('out'+n.zfill(3))
                    par_obj.gt_array.append(par_obj.folder_str+n.zfill(3)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+n.zfill(3)+'cell.png')
            


            if(j == 6):
                #dataset 6 SY Media.
                #dataset without challenging artefacts
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/final_data/data06-20130704-SY/'
                par_obj.csvPath = '/Users/dwaithe/Documents/collaborators/PiperM/analysis/'
                #par_obj.csv_file_name = 'imagesRequired-data6.csv'
                par_obj.csv_file_name = 'standard-data6.csv'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.file_str_arr = []
                par_obj.inc_bias_corr = True
                par_obj.train_at_each_itr = False
                par_obj.left_2_calc = np.arange(0,par_obj.num_of_im)
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    par_obj.file_str_arr.append('out'+n.zfill(3))
                    par_obj.gt_array.append(par_obj.folder_str+n.zfill(3)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+n.zfill(3)+'cell.png')
            

            if(j == 7):
                #dataset 7 SY Media.
                #Original dataset without challenging artefacts
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/final_data/data07-20130709-SY/'
                par_obj.csvPath = '/Users/dwaithe/Documents/collaborators/PiperM/analysis/'
                #par_obj.csv_file_name = 'imagesRequired-data7.csv'
                par_obj.csv_file_name = 'standard-data7.csv'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.file_str_arr = []
                par_obj.inc_bias_corr = True
                par_obj.train_at_each_itr = False
                par_obj.left_2_calc = np.arange(0,par_obj.num_of_im)
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    par_obj.file_str_arr.append('out'+n.zfill(3))
                    par_obj.gt_array.append(par_obj.folder_str+n.zfill(3)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+n.zfill(3)+'cell.png')
            if(j == 8):
               #dataset 8 SY Media.
                #Original dataset without challenging artefacts
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/final_data/data08-20140409-SY/'
                par_obj.csvPath = '/Users/dwaithe/Documents/collaborators/PiperM/analysis/'
                #par_obj.csv_file_name = 'imagesRequired-data8.csv'
                par_obj.csv_file_name = 'standard-data8.csv'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.file_str_arr = []
                par_obj.inc_bias_corr = True
                par_obj.train_at_each_itr = False
                par_obj.left_2_calc = np.arange(0,par_obj.num_of_im)
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    par_obj.file_str_arr.append('out'+n.zfill(3))
                    par_obj.gt_array.append(par_obj.folder_str+n.zfill(3)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+n.zfill(3)+'cell.png')
            if j == 9:
                #dataset 3 DM Media.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/final_data/data09-20140409-SY/'
                par_obj.csvPath = '/Users/dwaithe/Documents/collaborators/PiperM/analysis/'
                #par_obj.csv_file_name = 'imagesRequired-data3.csv'
                par_obj.csv_file_name = 'standard-data9.csv'
                
                par_obj.num_of_im = 8
                par_obj.left_2_calc = np.arange(0,par_obj.num_of_im)
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.file_str_arr = []
                par_obj.inc_bias_corr = True
                par_obj.train_at_each_itr = False
                #par_obj.TYPE_ROI = 'segRandIm'
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    par_obj.file_str_arr.append('out'+n.zfill(3))
                    par_obj.gt_array.append(par_obj.folder_str+n.zfill(3)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+n.zfill(3)+'cell.png')
            
            
            
            if j == 10:
                #dataset 10 SY Media.
                #purposefully selected images to express as large a range as possible.
                #For exemplifying non-artefact bias.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../final_data/data10-bias-SY/'
                par_obj.csvPath = '../analysis/'
                #par_obj.csv_file_name = 'imagesRequired-data10.csv'
                par_obj.csv_file_name = 'standard-data10.csv'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.inc_bias_corr = True
                par_obj.train_at_each_itr = False
                par_obj.left_2_calc = np.arange(0,par_obj.num_of_im)
                par_obj.frames_2_load = [[0],[0],[0],[0],[0],[0],[0],[0]]

                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    par_obj.file_str_arr.append('out'+n.zfill(3))
                    par_obj.gt_array.append(par_obj.folder_str+n.zfill(3)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+n.zfill(3)+'cell.png')









            if j == 11:
                #dataset 9: This was the data used for showing the tunability of the algorithm.
                #This is the Inclusive data.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140331/xiaoli/comp/easy/'
                par_obj.num_of_im = 5
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    file_str = 'V-'+n.zfill(1)
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str+'edots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str+'e.png')
            if j == 12:
                #dataset 9: This was the data used for showing the tunability of the algorithm.
                #This is the medium data. We don't include this in the final report.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140331/xiaoli/comp/medium/'
                par_obj.num_of_im = 5
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    file_str = 'V-'+n.zfill(1)
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str+'mdots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str+'m.png')
            if j == 13:
                #dataset 9: This was the data used for showing the tunability of the algorithm.
                #This is the Well-defined. data. We don't include this in the final report.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140331/xiaoli/comp/harsh/'
                par_obj.num_of_im = 5
                par_obj.test_im_start = 0
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    file_str = 'V-'+n.zfill(1)
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str+'hdots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str+'h.png')

            

            
            if j == 14:
                #dataset 1: defined Media.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140410/yeast/'
                par_obj.num_of_im = 5
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]

                file_str_index = ['yeast-00', 'yeast-01', 'yeast-02', 'yeast-03', 'yeast-04']
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = file_str_index[i]
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str_index[i]+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str_index[i]+'.png')
            if j == 15:
                #dataset 14: SY Media. Non-Artefact.
                #With controlled-lighting comparison
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140409/cathy/exp225/'
                par_obj.num_of_im = 8#10
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.inc_bias_corr = True
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = 'exp225-'+n.zfill(2)
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str[:-3]+'-'+n.zfill(2)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str+'_b.png')
            if j == 16:
                #dataset 15: SY Media. Artefact containing.s
                #With controlled-lighting comparison
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140409/cathy/exp225/'
                par_obj.num_of_im = 8#10
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.inc_bias_corr = True
                file_str_index = ['exp225-11', 'exp225-12', 'exp225-13', 'yeast-00', 'yeast-01', 'yeast-02', 'yeast-03', 'yeast-04','exp225-21', 'exp225-23']
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = file_str_index[i]
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str_index[i]+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str_index[i]+'_a.png')
            if j == 17:
                #dataset 1: defined Media. Non-artefact containing.
                #With controlled-lighting comparison
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140331/xiaoli/easy/'
                par_obj.num_of_im = 8#10
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.inc_bias_corr = True

                file_str_index = ['50N-1', '50N-2', '50N-3', '50N-4', '50N-5', '100N-1', '100N-2', '100N-3', '100N-4','100N-5']
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = file_str_index[i]
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str_index[i]+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str_index[i]+'.png')
            if j == 18:
                #dataset 17: defined Media. Artefact containing.
                #With controlled-lighting comparison
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140331/xiaoli/hard/'
                par_obj.num_of_im = 8#10
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.inc_bias_corr = True
                file_str_index = ['I-5', 'M-4', 'R-1', 'R-2', 'R-3', 'V-1', 'V-2', 'V-3', 'V-5', 'W-1']
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = file_str_index[i]
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str_index[i]+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str_index[i]+'.png')
            if j == 19:
                #dataset 18: defined Media. Iterative comparison.
                #Without controlled-lighting comparison
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/final_data/data08-20140331-DM/'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.TYPE_ROI = 'segRandIm'
                par_obj.inc_bias_corr = True
                file_str_index = [ '50N-1', '50N-2', '50N-3', '50N-4', '50N-5', '100N-1', '100N-2', '100N-3', '100N-4', '100N-5']
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = file_str_index[i]
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str_index[i]+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str_index[i]+'.png')
            if j == 20:
                #dataset 19: SY Media. Iterative comparison.
                #With controlled-lighting comparison
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140409/cathy/exp225/'
                par_obj.num_of_im = 10
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.TYPE_ROI = 'segRandIm'
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = 'exp225-'+n.zfill(2)
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str[:-3]+'-'+n.zfill(2)+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str+'.png')
            if j == 21:
                #dataset 20: SY Media. Iterative comparison.
                #With controlled-lighting comparison
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140409/cathy/exp225/'
                par_obj.num_of_im = 10
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.TYPE_ROI = 'segRandIm'
                par_obj.inc_bias_corr = True
                file_str_index = ['exp225-11', 'exp225-12', 'exp225-13', 'yeast-00', 'yeast-01', 'yeast-02', 'yeast-03', 'yeast-04','exp225-21', 'exp225-23']
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = file_str_index[i]
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str_index[i]+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str_index[i]+'_a.png')
            
            if j == 22:
                #dataset 22: defined Media.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/SY-handcount/'
                par_obj.num_of_im = 5
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                
                file_str_index = ['2h/2h-00', '2h/2h-03', '5h/5h-00', '5h/5h-01', 'ON/ON-04']
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = file_str_index[i]
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str_index[i]+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str_index[i]+'.png')

            if j == 23:
                #dataset 23: defined Media.
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/SY-handcount/'
                par_obj.num_of_im = 5
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                
                file_str_index = ['ON/ON-05', 'ON/ON-03', 'ON/ON-00', 'ON/ON-01', 'ONE/ONE-01', 'ONE/ONE-02']
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = file_str_index[i]
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str_index[i]+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str_index[i]+'.png')
            if j == 24:
                #dataset 24: CLUMPING SY media
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/UCL/project/data/clumping/'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.inc_bias_corr = True
                
                file_str_index = ['04_004','04_005', '04_006', '04_007', '09_001', '09_003', '09_005','09_008']
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = file_str_index[i]
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str_index[i]+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str_index[i]+'cell.png')
            if j == 25:
                #dataset 22: Non-CLUMPING SY media
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/UCL/project/data/nonclumping/'
                par_obj.num_of_im = 8
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.inc_bias_corr = True
                
                file_str_index = ['04_001', '04_002', '04_003', '09_006', '04_008', '09_002','09_004','09_007']
                for i in range(0, par_obj.num_of_im):
                    n = str(i)
                    file_str = file_str_index[i]
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str_index[i]+'dots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str_index[i]+'cell.png')
            if j == 26:
                
                #This is the Inclusive data.
                #dataset 1: transparent DM media
                #Original dataset without challenging artefacts
                par_obj.sigma_data = 1.0
                par_obj.feature_scale = 0.8
                par_obj.folder_str = '../../../../../Documents/collaborators/PiperM/20140331/xiaoli/comp/easy/'
                par_obj.num_of_im = 5
                par_obj.test_im_start = []
                par_obj.test_im_end = []
                par_obj.file_str_arr = []
                par_obj.file_ext = 'png'
                par_obj.ch_active = [0, 1, 2]
                par_obj.CC =[]
                par_obj.CC_absErr =[]
                par_obj.CC_perErr =[]
                par_obj.cross_valid_method = 'leaveOneOut'
                for i in range(1, par_obj.num_of_im+1):
                    n = str(i)
                    file_str = 'V-'+n.zfill(1)
                    par_obj.file_str_arr.append(file_str)
                    par_obj.gt_array.append(par_obj.folder_str+file_str+'edots.png')
                    par_obj.file_array.append(par_obj.folder_str+file_str+'e.png')
            
                


            if(par_obj.cross_valid_method == 'leaveOneOut'):
                par_obj.num_of_train_im = par_obj.num_of_im-1
                #Number of train images.
                par_obj.num_of_iterations = par_obj.num_of_train_im
                par_obj.num_of_experiments = par_obj.num_of_im
                #Range of train and test Images.
                par_obj.test_im_start = 0
                
                par_obj.test_im_end = par_obj.num_of_im
                par_obj.iter_to_train_forest = [par_obj.num_of_train_im-1]
            if(par_obj.cross_valid_method == 'full'):
                par_obj.num_of_train_im = par_obj.num_of_im
                #Number of train images.
                par_obj.num_of_iterations = par_obj.num_of_train_im
                par_obj.num_of_experiments = par_obj.num_of_im
                #Range of train and test Images.
                par_obj.test_im_start = 0
                par_obj.train_at_each_itr = True
                par_obj.test_im_end = par_obj.num_of_im
                par_obj.cross_valid_method = False
                par_obj.iter_to_train_forest = [par_obj.num_of_train_im]

            if(par_obj.TYPE_ROI == 'randomROI' or par_obj.TYPE_ROI == 'segActiveDen' or par_obj.TYPE_ROI =='segRandIm'):
                #In this mode iterations correspond to regions.
                par_obj.num_of_iterations = 500
                par_obj.train_at_each_itr = False
                #At which iterations to train forest if above set to false, starts at 0.
                par_obj.iter_to_train_forest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,22,23,24,25,26,27,28,29, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 300, 400, 500, 600, 700]

            save_parameters()
            #features are refreshed. once per experiment.
            par_obj.feat_arr = {}
            for v in range(0, 5):
                
                output_header()
                #The number of iterations to run.
                for i in range(0, par_obj.num_of_experiments):
                    par_obj.curr_experiment = i
                    #The training dataset.
                    if(par_obj.cross_valid_method == False):
                        par_obj.im_num_range = np.random.choice(range(par_obj.test_im_start, par_obj.test_im_end), size=par_obj.num_of_train_im, replace=par_obj.replace_im)
                    


                    
                    #Creates empty array to record density estimation.
                    #Makes sure no data is inherited from one experiment to the next.
                    par_obj.dots_count_arr = [0]*par_obj.num_of_im
                    par_obj.dots = []
                    par_obj.rects = np.zeros((1, 4))
                    par_obj.saved_dots = []
                    par_obj.saved_ROI = []
                    par_obj.var = []
                    par_obj.subdivide_ROI = []
                    par_obj.first_time = True
                    par_obj.dense_array = []
                    par_obj.dots_array = []
                    par_obj.to_process =[]
                    par_obj.total_count_dots = 0
                    

                    v2.import_data_fn(par_obj, par_obj.file_array)

                    for b in range(par_obj.test_im_start, par_obj.test_im_end):
                        im_array = np.zeros((par_obj.height, par_obj.width))
                        par_obj.dense_array.append(im_array)
                        par_obj.dots_array.append(im_array)
                    if(par_obj.cross_valid_method =='leaveOneOut'):
                        par_obj.left_2_calc = range(par_obj.test_im_start, par_obj.test_im_end)
                        par_obj.left_2_calc.pop(i)
                        par_obj.left_2_calc= np.random.choice(par_obj.left_2_calc, size=par_obj.num_of_train_im-1, replace=par_obj.replace_im)
                    
                        par_obj.test_im_start = i
                        par_obj.test_im_end = i+1

                    print('image range:' +str(par_obj.left_2_calc))
                    print('test range:'+str(par_obj.test_im_start)+str(par_obj.test_im_end))

                    auto_fn()
                    #output_results()
