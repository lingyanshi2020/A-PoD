#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import*
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import random
from skimage import io
from scipy.interpolate import splprep, splev
import cv2
from scipy import misc
from scipy import ndimage
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy import stats
from skimage.transform import resize
import matplotlib.pyplot as plt
import math
import gc
import os
import pandas as pd
import time


# In[2]:


root = Tk() #the base for origional creation of the window
root.title("3D A-PoD GUI")
root.geometry("230x430")


# In[3]:


def PSF1():
    global inputpsf1 
    inputpsf1 = filedialog.askopenfilename(filetypes=(("tif files","*.tif"),("all files","*.*")))


# In[4]:


def PSF2():
    global inputpsf2
    inputpsf2 = filedialog.askopenfilename(filetypes=(("tif files","*.tif"),("all files","*.*")))
    


# In[5]:

def inputImage():
    global inputImagee
    inputImagee = filedialog.askopenfilename(filetypes=(("tif files", "*.tif"), ("all files", "*.*")))
    filee = os.path.split(inputImagee)
    global fileaddr
    fileaddr = filee[0]+'/'
    global filename
    fullname = filee[1]
    filename = fullname[0:(len(fullname)-4)]


# In[6]:


def run():
    #fileaddr = 'C:/Hongje_data/'
    #filename = 'sat-lipid_bicubic'

    #PSF_folder = 'C:/Hongje_data/PSF/'
    PSF_1 = inputpsf1
    PSF_2 = inputpsf2

    num_spot = float(num1.get())
    numbmaxaddr = float(num2.get())
    l_rate_x = float(num31.get())
    l_rate_z = float(num32.get())
    th_value = float(num4.get())
    reblur_factor = float(num5.get())
    
    psfsize = 100

    def z_extrapolimg(image, psf):
    
        z_xtrapolnum = 3
    
        max_addr = np.where(psf == np.max(psf))
    
        newimg = np.zeros([image.shape[0]+2*(z_xtrapolnum), image.shape[1], image.shape[2]])
    
        newimg[z_xtrapolnum:z_xtrapolnum+image.shape[0], :, :] = image
    
        Inten_ratio = psf[max_addr[0][0]:max_addr[0][0]+z_xtrapolnum, max_addr[1][0], max_addr[2][0]]/psf[max_addr[0][0], max_addr[1][0], max_addr[2][0]]
    
        for i in range(1, z_xtrapolnum-1):
        
            newimg[z_xtrapolnum+image.shape[0]+i-1, :, :] = image[image.shape[0]-1, :, :]*Inten_ratio[i-1]*0.1
            newimg[z_xtrapolnum-i, :, :] = image[0, :, :]*Inten_ratio[i-1]*0
        
        return newimg

    def posimage(image):
    
        addr = np.where(image < 0)
        for i in range(0, addr[0].size):
            image[addr[0][i], addr[1][i], addr[2][i]] = 0
    
        return image

    def filtimage(image):
    
        filtimage = np.ones(image.shape)
    
        addr = np.where(image < 0)
        for i in range(0, addr[0].size):
            filtimage[addr[0][i], addr[1][i], addr[2][i]] = 0
    
        return filtimage*100

    def lp_filt(image, psf, imsize):
    
        fftpsf = np.fft.fftn(psf, s=imsize)
        fftimage = np.fft.fftn(image, s=imsize)
    
        OTF_filt = np.zeros(fftimage.shape)
    
        pos_addr = np.where(np.absolute(fftpsf) > np.max(np.absolute(fftpsf))*0.01)
        OTF_filt[pos_addr] = 1
    
        newfftimage = OTF_filt * fftimage
    
        newimage = np.fft.ifftn(newfftimage, s=imsize)
    
        return newimage

    def lp_filt_2(image, psf, imsize):
    
        fftpsf = np.fft.fftn(psf, s=imsize)
        fftimage = np.fft.fftn(image, s=imsize)
    
        OTF_filt = np.zeros(fftimage.shape)
    
        pos_addr = np.where(np.absolute(fftpsf) > np.max(np.absolute(fftpsf))*0.1)
        OTF_filt[pos_addr] = 1
    
        newfftimage = OTF_filt * fftimage
    
        newimage = np.fft.ifftn(newfftimage, s=imsize)
    
        return newimage

    def SNR_cal(a, psf, imsize): 
    
        img_noise = hp_filt(a, psf, imsize)
        img_signal = lp_filt(a, psf, imsize)
        img_offset = det_offset(a, psf, imsize)
        SNR = (np.mean(np.abs(img_signal)))/(np.std(img_noise))
    
        return SNR

        

    def hp_filt(image, psf, imsize):
    
        fftpsf = np.fft.fftn(psf, s=imsize)
        fftimage = np.fft.fftn(image, s=imsize)
    
        OTF_filt = np.ones(fftimage.shape)
    
        pos_addr = np.where(np.absolute(fftpsf) > np.max(np.absolute(fftpsf))*0.01)
        OTF_filt[pos_addr] = 0
    
        newfftimage = OTF_filt * fftimage
    
        newimage = np.fft.ifftn(newfftimage, s=imsize)
    
        return newimage

    def det_offset(image, psf, imsize):
    
        fftpsf = np.fft.fftn(psf, s=imsize)
        fftimage = np.fft.fftn(image, s=imsize)
    
        OTF_filt = np.ones(fftimage.shape)
    
        pos_addr = np.where(np.absolute(fftpsf) < np.max(np.absolute(fftpsf))*0.90)
        OTF_filt[pos_addr] = 0
    
        newfftimage = OTF_filt * fftimage
    
        newimage = np.fft.ifftn(newfftimage, s=imsize)
    
        return newimage



    def psf_crop(psf):
    
    
        max_addr = np.where(psf == np.max(psf))
    
    
        max_val_psf = psf[max_addr[0][0], max_addr[1][0], max_addr[2][0]]
    
        max_val_psf_z = psf[0:psf.shape[0], max_addr[1][0], max_addr[2][0]]
        max_val_psf_x = psf[max_addr[0][0], max_addr[1][0], 0:psf.shape[2]]
    
    
        for i in range(1, psf.shape[0]):
        
            val_psf_z = psf[max_addr[0][0]+i, max_addr[1][0], max_addr[2][0]]
        
            if val_psf_z < 0.5*max_val_psf:
            
                psf_size[0] = ((i)*3)
                break
    
        for j in range(1, psf.shape[1]):
        
            val_psf_x = psf[max_addr[0][0], max_addr[1][0], max_addr[2][0]+j]
        
            if val_psf_x < 0.5*max_val_psf:
            
                psf_size[1] = ((j)*3)
                break
    
        new_psf = psf[max_addr[0][0]-psf_size[0]:max_addr[0][0]+psf_size[0]+1, max_addr[1][0]-psf_size[1]:max_addr[1][0]+psf_size[1]+1, max_addr[2][0]-psf_size[1]:max_addr[2][0]+psf_size[1]+1]
    
    
        return new_psf, psf_size

        
    def virtualimage(imsize, addr_z, addr_x, addr_y, sizeaddr):
    
        tempimg = np.zeros(imsize[0]*imsize[1]*imsize[2])
    
        unit_inten = 1
    
    
        intarray = np.ones(sizeaddr)*unit_inten

        addr_z = tf.where(addr_z > imsize[0]-1, tf.constant(0, dtype=tf.float32), addr_z)
        addr_z = tf.where(addr_z < 0, tf.constant(imsize[0]-1, dtype=tf.float32), addr_z)
        addr_x = tf.where(addr_x > imsize[1]-1, tf.constant(0, dtype=tf.float32), addr_x)
        addr_x = tf.where(addr_x < 0, tf.constant(imsize[1]-1, dtype=tf.float32), addr_x)
        addr_y = tf.where(addr_y > imsize[2]-1, tf.constant(0, dtype=tf.float32), addr_y)
        addr_y = tf.where(addr_y < 0, tf.constant(imsize[2]-1, dtype=tf.float32), addr_y)
        
        new_address = addr_y + addr_x*imsize[2] + addr_z*imsize[2]*imsize[1]
    
        int_address = tf.math.round(new_address)
        int_address = tf.cast(int_address, dtype=tf.int32)
        int_address = tf.clip_by_value(int_address, 0, imsize[0]*imsize[1]*imsize[2]-1)
        tensorintarray = tf.cast(intarray, dtype=tf.float32)
        img_var = tf.constant(tempimg, dtype=tf.float32)
    
        new_img_var = tf.tensor_scatter_nd_add(img_var, int_address, tensorintarray)
        new_img_var = tf.cast(new_img_var, dtype=tf.float32)

        del [[tempimg, int_address, tensorintarray, img_var]]
    
        return new_img_var
    
    def convimggen(imsize, temp_addr_z, temp_addr_x, temp_addr_y, psf, sizeaddr):
    
        temp_psf = tf.constant(psf.reshape(psf.shape[0], psf.shape[1], psf.shape[2], 1, 1), dtype=tf.float32)
        tf.reshape(virtualimage(imsize, temp_addr_z, temp_addr_x, temp_addr_y, addrsize), [imsize[0], imsize[1], imsize[2]])
    
        temp_image = virtualimage(imsize, temp_addr_z, temp_addr_x, temp_addr_y, sizeaddr)
        temp_image = tf.reshape(temp_image, [1, imsize[0], imsize[1], imsize[2], 1])
        temp_convimg = tf.nn.conv3d(temp_image, temp_psf, strides=[1, 1, 1, 1, 1], padding='SAME')
    
        re_temp_convimg = tf.math.reduce_sum(temp_psf)*tf.math.divide(tf.reshape(temp_convimg, [imsize[0], imsize[1], imsize[2]]),tf.math.reduce_sum(temp_convimg))*sizeaddr
    
        del [[temp_psf, temp_image, temp_convimg]]
    
    
        return re_temp_convimg


    def cost_fun(re_image, imsize, temp_addr_x, temp_addr_y, psf, sizeaddr):
    
        cost = tf.nn.l2_loss(tf.math.subtract(re_image, convimggen(imsize, temp_addr_x, temp_addr_y, psf, sizeaddr)))
    
        return cost


    def adamIntVar(re_image, imsize, temp_addr_z, temp_addr_x, temp_addr_y, psf, sizeaddr, m_z, v_z, m_x, v_x, m_y, v_y, extrapolnum, i, imagefilter):
    
        LR_z = tf.constant(psf_size[0]*l_rate_z, dtype=tf.float32)
        LR_xy = tf.constant(psf_size[1]*l_rate_x, dtype=tf.float32)
    
    
        beta_1 = tf.constant(0.9, dtype=tf.float32)
        beta_1_1 = 1-beta_1
        beta_2 = tf.constant(0.99, dtype=tf.float32)
        beta_2_2 = 1-beta_2
    
        m_z_new = tf.cast(m_z, dtype=tf.float32)
        v_z_new = tf.cast(v_z, dtype=tf.float32)

        m_x_new = tf.cast(m_x, dtype=tf.float32)
        v_x_new = tf.cast(v_x, dtype=tf.float32)
    
        m_y_new = tf.cast(m_y, dtype=tf.float32)
        v_y_new = tf.cast(v_y, dtype=tf.float32)
    
        z_res = temp_addr_z
        x_res = temp_addr_x
        y_res = temp_addr_y
    
        grad_res_z, grad_res_x, grad_res_y = gradgenfun(re_image, imsize, temp_addr_z, temp_addr_x, temp_addr_y, psf, sizeaddr, imagefilter)
    
        m_z_new = beta_1*m_z + beta_1_1*tf.cast(grad_res_z, dtype=tf.float32)
        v_z_new = beta_2*v_z + beta_2_2*tf.cast(tf.math.square(grad_res_z), dtype=tf.float32)
    
        z_res = temp_addr_z - LR_z*(m_z_new/(1-tf.math.pow(beta_1, i)))/tf.math.sqrt(v_z_new/(1-tf.math.pow(beta_2,i)) + tf.constant(0.001, dtype=tf.float32))
    
        z_res = tf.math.round(z_res)

        m_x_new = beta_1*m_x + beta_1_1*tf.cast(grad_res_x, dtype=tf.float32)
        v_x_new = beta_2*v_x + beta_2_2*tf.cast(tf.math.square(grad_res_x), dtype=tf.float32)
    
        x_res = temp_addr_x - LR_xy*(m_x_new/(1-tf.math.pow(beta_1, i)))/tf.math.sqrt(v_x_new/(1-tf.math.pow(beta_2,i)) + tf.constant(0.001, dtype=tf.float32))
    
        x_res = tf.math.round(x_res)
    
        m_y_new = beta_1*m_y + beta_1_1*tf.cast(grad_res_y, dtype=tf.float32)
        v_y_new = beta_2*v_y + beta_2_2*tf.cast(tf.math.square(grad_res_y), dtype=tf.float32)
    
        y_res = temp_addr_y - LR_xy*(m_y_new/(1-tf.math.pow(beta_1, i)))/tf.math.sqrt(v_y_new/(1-tf.math.pow(beta_2,i)) + tf.constant(0.001, dtype=tf.float32))
    
        y_res = tf.math.round(y_res)
    
    
        new_z_res = tf.cast(z_res, dtype=tf.float32)
        new_x_res = tf.cast(x_res, dtype=tf.float32)
        new_y_res = tf.cast(y_res, dtype=tf.float32)
        
        new_z_res = tf.where(new_z_res > imsize[0]-1, tf.constant(0, dtype=tf.float32), new_z_res)
        new_z_res = tf.where(new_z_res < 0, tf.constant(imsize[0]-1, dtype=tf.float32), new_z_res)
        new_x_res = tf.where(new_x_res > imsize[1]-1, tf.constant(0, dtype=tf.float32), new_x_res)
        new_x_res = tf.where(new_x_res < 0, tf.constant(imsize[1]-1, dtype=tf.float32), new_x_res)
        new_y_res = tf.where(new_y_res > imsize[2]-1, tf.constant(0, dtype=tf.float32), new_y_res)
        new_y_res = tf.where(new_y_res < 0, tf.constant(imsize[2]-1, dtype=tf.float32), new_y_res)
    
    
        diff_sum = tf.math.abs(tf.math.reduce_sum(tf.math.abs(tf.math.subtract(convimggen(imsize, temp_addr_z, temp_addr_x, temp_addr_y, psf, sizeaddr), re_image))) - tf.math.reduce_sum(tf.math.abs(tf.math.subtract(re_image, convimggen(imsize, new_z_res, new_x_res, new_y_res, psf, sizeaddr)))))
    
        #del [[Less_addr_x, Less_addr_y, Greater_addr_x, Greater_addr_y, Less_Tensor_x, Less_Tensor_y, Greater_Tensor_x, Greater_Tensor_y]]
    
    
        return new_z_res, new_x_res, new_y_res, m_z_new, v_z_new, m_x_new, v_x_new, m_y_new, v_y_new, diff_sum


    
    def gradgenfun(re_image, imsize, temp_addr_z, temp_addr_x, temp_addr_y, psf, sizeaddr, imagefilter):
    
        grad_step = tf.constant(1, dtype=tf.int32)
    
        temp_result =  1000*tf.math.subtract(re_image, convimggen(imsize, temp_addr_z, temp_addr_x, temp_addr_y, psf, sizeaddr))
    
        result_z_1_1= tf.roll(temp_result, shift=[1, 0, 0], axis=[0, 0, 1])
        result_z_1_2= tf.roll(temp_result, shift=[grad_step, 0, 0], axis=[0, 0, 1])
        result_z_1_3= tf.roll(temp_result, shift=[2*grad_step, 0, 0], axis=[0, 0, 1])
        result_z_1_4= tf.roll(temp_result, shift=[3*grad_step, 0, 0], axis=[0, 0, 1])
        result_z_2_1= tf.roll(temp_result, shift=[-1, 0, 0], axis=[0, 0, 1])
        result_z_2_2= tf.roll(temp_result, shift=[-1*grad_step, 0, 0], axis=[0, 0, 1])
        result_z_2_3= tf.roll(temp_result, shift=[-2*grad_step, 0, 0], axis=[0, 0, 1])
        result_z_2_4= tf.roll(temp_result, shift=[-3*grad_step, 0, 0], axis=[0, 0, 1])
    
    
        result_z_1 = (result_z_1_1 + result_z_1_2 + result_z_1_3 + result_z_1_4)
        result_z_2 = (result_z_2_1 + result_z_2_2 + result_z_2_3 + result_z_2_4)

        result_x_1_1= tf.roll(temp_result, shift=[0, 1, 0], axis=[0, 0, 1])
        result_x_1_2= tf.roll(temp_result, shift=[0, grad_step, 0], axis=[0, 0, 1])
        result_x_1_3= tf.roll(temp_result, shift=[0, 2*grad_step, 0], axis=[0, 0, 1])
        result_x_1_4= tf.roll(temp_result, shift=[0, 3*grad_step, 0], axis=[0, 0, 1])
        result_x_2_1= tf.roll(temp_result, shift=[0, -1, 0], axis=[0, 0, 1])
        result_x_2_2= tf.roll(temp_result, shift=[0, -1*grad_step, 0], axis=[0, 0, 1])
        result_x_2_3= tf.roll(temp_result, shift=[0, -2*grad_step, 0], axis=[0, 0, 1])
        result_x_2_4= tf.roll(temp_result, shift=[0, -3*grad_step, 0], axis=[0, 0, 1])
    
    
        result_x_1 = (result_x_1_1 + result_x_1_2 + result_x_1_3 + result_x_1_4)
        result_x_2 = (result_x_2_1 + result_x_2_2 + result_x_2_3 + result_x_2_4)
    
        result_y_1_1= tf.roll(temp_result, shift=[0, 0, 1], axis=[0, 0, 1])
        result_y_1_2= tf.roll(temp_result, shift=[0, 0, grad_step], axis=[0, 0, 1])
        result_y_1_3= tf.roll(temp_result, shift=[0, 0, 2*grad_step], axis=[0, 0, 1])
        result_y_1_4= tf.roll(temp_result, shift=[0, 0, 3*grad_step], axis=[0, 0, 1])
        result_y_2_1= tf.roll(temp_result, shift=[0, 0, -1], axis=[0, 0, 1])
        result_y_2_2= tf.roll(temp_result, shift=[0, 0, -1*grad_step], axis=[0, 0, 1])
        result_y_2_3= tf.roll(temp_result, shift=[0, 0, -2*grad_step], axis=[0, 0, 1])
        result_y_2_4= tf.roll(temp_result, shift=[0, 0, -3*grad_step], axis=[0, 0, 1])
    
    
        result_y_1 = (result_y_1_1 + result_y_1_2 + result_y_1_3 + result_y_1_4)
        result_y_2 = (result_y_2_1 + result_y_2_2 + result_y_2_3 + result_y_2_4)
    
        temp_addr_z = tf.where(temp_addr_z > imsize[0]-1, tf.constant(0, dtype=tf.float32), temp_addr_z)
        temp_addr_z = tf.where(temp_addr_z < 0, tf.constant(imsize[0]-1, dtype=tf.float32), temp_addr_z)
        temp_addr_x = tf.where(temp_addr_x > imsize[1]-1, tf.constant(0, dtype=tf.float32), temp_addr_x)
        temp_addr_x = tf.where(temp_addr_x < 0, tf.constant(imsize[1]-1, dtype=tf.float32), temp_addr_x)
        temp_addr_y = tf.where(temp_addr_y > imsize[2]-1, tf.constant(0, dtype=tf.float32), temp_addr_y)
        temp_addr_y = tf.where(temp_addr_y < 0, tf.constant(imsize[2]-1, dtype=tf.float32), temp_addr_y)
    
        result_img_z = tf.math.subtract(result_z_1, result_z_2)
        result_z = tf.gather_nd(result_img_z, tf.cast(tf.concat([temp_addr_z, temp_addr_x, temp_addr_y], 1), dtype = tf.int32))
        result_z = tf.reshape(tf.cast(result_z, dtype=tf.float32), [sizeaddr, 1])

        result_img_x = tf.math.subtract(result_x_1, result_x_2)
        result_x = tf.gather_nd(result_img_x, tf.cast(tf.concat([temp_addr_z, temp_addr_x, temp_addr_y], 1), dtype = tf.int32))
        result_x = tf.reshape(tf.cast(result_x, dtype=tf.float32), [sizeaddr, 1])
    
        result_img_y = tf.math.subtract(result_y_1, result_y_2)
        result_y = tf.gather_nd(result_img_y, tf.cast(tf.concat([temp_addr_z, temp_addr_x, temp_addr_y], 1), dtype = tf.int32))
        result_y = tf.reshape(tf.cast(result_y, dtype=tf.float32), [sizeaddr, 1])
    
        return result_z, result_x, result_y

    def clearall():
        all = [var for var in globals() if var[0] != "_"]
    
        for var in all:
            del globals()[var]
        
    def address_exporter(finalres_z, finalres_x, finalres_y):
        temp_res_z = finalres_z
        temp_res_x = finalres_x
        temp_res_y = finalres_y
        return temp_res_z, temp_res_x, temp_res_y


    def intensity_ratio(convimage, dotimage, jj, iternumb, addrsize):
    
        for jj in range(iternumb+1):
        
            testimage = posimage(jj*dotimage - convimage)
        
               
            if np.sum(testimage)>0:
            
                break
            
        return jj


    #####################

    ex_psf = io.imread(PSF_1)
    em_psf = io.imread(PSF_2)

    psf = posimage(np.sqrt(ex_psf*em_psf))

    del [[ex_psf, em_psf]]

    psf_size = [0, 0]
    psf = psf/np.sum(psf)
    new_psf, psf_size = psf_crop(psf)

    psf_ratio = np.sum(new_psf)

    big_psf = new_psf

    big_psf = big_psf/np.sum(big_psf)


    #######################

    start = time.time()


    intratio = 1

    image = np.float32(io.imread(inputImagee))


    init_sum_val = np.sum(image)
    
    #bground = np.min(image) + th_value*np.std(image)
    bground = th_value


    image = posimage(image-bground)
    sum_val_image = np.sum(image)
    
    max_val = np.max(image)
    
    thrval = np.mean(image) + 1*np.std(image)

    addr = np.where(image > thrval)
    [numberofaxes, tempsize] = np.shape(addr)
    
    intensityratio = np.sum(posimage(image - 0)/np.sum(image))

    totaladdrsize = np.var(image)*tempsize/(np.mean(image)+1)/(np.mean(image)+1)*num_spot
    
    imagefilter = filtimage(image - thrval)
    io.imsave(filename+'_imagefilter.tif', np.float32(imagefilter))

    image = posimage(image)


    if totaladdrsize <= numbmaxaddr:
    
        addrsize = np.int(np.floor(totaladdrsize))
        iternumb = 1
    
    else:
    
        iternumb = np.int(np.floor((totaladdrsize/numbmaxaddr)))
        addrsize = np.int(np.floor((totaladdrsize/iternumb)))

    intratio = np.sum(np.abs(posimage(image-thrval)))/iternumb/addrsize
    
    big_psf = big_psf/np.sum(big_psf)
    
    real_psf = psf_ratio*big_psf*intratio
    big_psf = psf_ratio*big_psf*intratio
    
    
    image_org = image
    io.imsave(filename+'_temp.tif', np.float32(image))
    
    imsize = [image.shape[0], image.shape[1], image.shape[2], intratio]
    im_orgsize = [image_org.shape[0], image_org.shape[1], image_org.shape[2]]
    

    addr_z = np.zeros((addrsize, 1))
    addr_x = np.zeros((addrsize, 1))
    addr_y = np.zeros((addrsize, 1))
    
    addr = np.array(addr)
        
    #for i in range(addrsize):
    #
    #    randaddr = random.randint(0, addr.shape[1]-1)
    #    
    #    addr_x[i] = addr[0, randaddr]
    #    addr_y[i] = addr[1, randaddr]

    randaddr = np.random.randint(0, addr.shape[1]-1, size=addrsize)
    addr_z = np.reshape(addr[0, randaddr], [addrsize, 1])
    addr_x = np.reshape(addr[1, randaddr], [addrsize, 1])
    addr_y = np.reshape(addr[2, randaddr], [addrsize, 1])
    
    io.imsave('temp_imsave.tif', np.zeros((im_orgsize[0], im_orgsize[1], im_orgsize[2])))
    io.imsave('temp_imresidue.tif', np.float32(posimage(image)))
    

    
    gc.enable()
    
    jj = 0
    
    SNR = SNR_cal(image, psf, image.shape)
    
    while (iternumb - jj)/iternumb > 0.01:

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
          # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
          try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15360)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

        image = io.imread('temp_imresidue.tif')
        simage = io.imread('temp_imsave.tif')
        
        SNR = SNR_cal(image, big_psf, image.shape)
        
        image = posimage(image)
               
        sum_image = tf.Variable(simage, "sum_image", dtype=tf.float32)
            
        residue = tf.Variable(image, "residue", dtype=tf.float32)
        
        extrapolnum = np.float32(math.ceil(big_psf.shape[0]/2))

    
        
        #for i in range(addrsize):
        #
        #    randaddr = random.randint(0, addr.shape[1]-1)
        #
        #    addr_x[i] = addr[0, randaddr] + psf_size*random.uniform(-1, 1)
        #    addr_y[i] = addr[1, randaddr] + psf_size*random.uniform(-1, 1)

        randaddr = np.random.randint(0, addr.shape[1]-1, size=addrsize)
        rand_shift_z = psf_size[0] * np.random.uniform(low=-1, high=1, size=(1, addrsize))
        rand_shift_x = psf_size[1] * np.random.uniform(low=-1, high=1, size=(2, addrsize))
        addr_z = np.reshape(addr[0, randaddr] + rand_shift_z[0], [addrsize, 1])
        addr_x = np.reshape(addr[1, randaddr] + rand_shift_x[0], [addrsize, 1])
        addr_y = np.reshape(addr[2, randaddr] + rand_shift_x[1], [addrsize, 1])

        #randaddr = np.random.randint(0, addr.shape[1]-1, size=addrsize)
        #addr_x = addr[0, randaddr]
        #addr_y = addr[1, randaddr]
    
        finalres_z = tf.Variable(addr_z, dtype=tf.float32)
        finalres_x = tf.Variable(addr_x, dtype=tf.float32)
        finalres_y = tf.Variable(addr_y, dtype=tf.float32)
            
        
        if jj == 1:
            
            re_image = tf.cast(image, tf.float32)
                
        else:
            
            re_image = residue
                
        finalres_z = tf.cast(addr_z, dtype=tf.float32)
        finalres_x = tf.cast(addr_x, dtype=tf.float32)
        finalres_y = tf.cast(addr_y, dtype=tf.float32)
            
    
            
        m = tf.zeros(tf.shape(finalres_x), dtype=tf.float32)
        v = tf.zeros(tf.shape(finalres_x), dtype=tf.float32)
        i = tf.constant(1, dtype=tf.float32)
        
        m_z_new = m
        v_z_new = v
        m_x_new = m
        v_x_new = v
        m_y_new = m
        v_y_new = v
        
        j = tf.constant(1, dtype=tf.float32)
        
        diff_sum = tf.constant(1, dtype=tf.float32)
        
        iterthrval = intratio*addrsize*1e-4
            
        def cond(j, finalres_z, finalres_x, finalres_y, m_z_new, v_z_new, m_x_new, v_x_new, m_y_new, v_y_new, diff_sum, extrapolnum):
            
            return tf.math.logical_or(tf.less(j, 10), tf.greater_equal(diff_sum, iterthrval))
            
        def body(j, finalres_z, finalres_x, finalres_y, m_z_new, v_z_new, m_x_new, v_x_new, m_y_new, v_y_new, diff_sum, extrapolnum):
            j = tf.add(j, 1)
            finalres_z, finalres_x, finalres_y, m_z_new, v_z_new, m_x_new, v_x_new, m_y_new, v_y_new, diff_sum = adamIntVar(re_image, imsize, finalres_z, finalres_x, finalres_y, big_psf, addrsize, m_z_new, v_z_new, m_x_new, v_x_new, m_y_new, v_y_new, extrapolnum, j, imagefilter)
            return [j, finalres_z, finalres_x, finalres_y, m_z_new, v_z_new, m_x_new, v_x_new, m_y_new, v_y_new, diff_sum, extrapolnum]
        
        res_loop = tf.while_loop(cond, body, [j, finalres_z, finalres_x, finalres_y, m_z_new, v_z_new, m_x_new, v_x_new, m_y_new, v_y_new, diff_sum, extrapolnum])
        
        [j, finalres_z, finalres_x, finalres_y, m_z_new, v_z_new, m_x_new, v_x_new, m_y_new, v_y_new, diff_sum, extrapolnum] = res_loop
        
        resultconvimage = convimggen(imsize, finalres_z, finalres_x, finalres_y, real_psf, addrsize)
    
        resultconvimage = tf.reshape(resultconvimage, [imsize[0], imsize[1], imsize[2]])
        
        resultimage = tf.reshape(virtualimage(imsize, finalres_z, finalres_x, finalres_y, addrsize), [imsize[0], imsize[1], imsize[2]])
    
        sub_image = tf.cast(tf.math.subtract(re_image, resultconvimage), tf.float32)
        temp_image = tf.cast(sum_image, dtype=tf.float32)
        
        if jj == 1:
            
            sum_image = resultimage
            residue = sub_image
            
            
        else:
                
            sum_image = tf.cast(tf.add(temp_image, resultimage), tf.float32)
            residue = tf.cast(sub_image, tf.float32)
                
        [rimage, smallconvimage] = [resultimage.numpy(), resultconvimage.numpy()]
        
        orgimage = image[:, :, :]
        
        
        
        jj += 1
        
        orgimage = orgimage - smallconvimage
        
        image = orgimage
        
        io.imsave('temp_imresidue.tif', np.float32(image))
        io.imsave('temp_imresidue_pos.tif', posimage(np.float32(image)))

        
        io.imsave('temp_imsave.tif', np.float32(rimage+simage))
        
        
        print(jj/iternumb*100, "% processed, ", "time:", time.time() - start, "SNR:", SNR)
        #tf.keras.backend.clear_session()
    
        del [[simage, residue, temp_image, re_image, sub_image, sum_image, resultconvimage, resultimage, m_x_new, v_x_new, m_y_new, v_y_new, m, v, i, finalres_x, finalres_y]]
        
        gc.collect()
        
    simage = io.imread('temp_imsave.tif')
    
    rimage = io.imread('temp_imresidue.tif')
    
    f_rimage = intratio*simage + bground
    f_rimage = f_rimage/np.sum(f_rimage)*init_sum_val

    if reblur_factor > 0:

        f_rimage = gaussian_filter(f_rimage, sigma = reblur_factor)
    
    
    if not(os.path.isdir(fileaddr + 'Deconvolved')):
        os.makedirs(os.path.join(fileaddr + 'Deconvolved'))
    if not(os.path.isdir(fileaddr + 'Deconvolved_pos_residue')):
        os.makedirs(os.path.join(fileaddr + 'Deconvolved_pos_residue'))
    if not(os.path.isdir(fileaddr + 'Deconvolved_residue')):
        os.makedirs(os.path.join(fileaddr + 'Deconvolved_residue'))
    
    io.imsave(fileaddr + 'Deconvolved/'+filename+'_A-PoD.tif', np.float32(f_rimage))
    io.imsave(fileaddr + 'Deconvolved_pos_residue/'+filename+'_A-PoD_posresidue.tif', posimage(rimage))
    io.imsave(fileaddr + 'Deconvolved_residue/'+filename+'A-PoD_residue.tif', rimage/np.sum(rimage)*sum_val_image)
    
    print("elapsed time: ", (time.time() - start)/60, "min, SNR: ", SNR)
    
    
    del [[image, simage, rimage, jj, iternumb, thrval, totaladdrsize, addr, addr_x, addr_y]]
    
    gc.collect()
    


   


# In[ ]:





# In[ ]:



# In[ ]:

PSF1button = Button(root, text="First PSF",padx=70, pady=10,command=PSF1) 
PSF1button.grid(row=10, column=45)

PSF2button = Button(root, text ="Second PSF",padx=70, pady=10, command=PSF2)
PSF2button.grid(row=15, column=45)

inputImagebutton = Button(root, text="Original image",padx=70, pady=10, command=inputImage)
inputImagebutton.grid(row=20, column=45)

runbutton = Button(root, text="RUN",padx=50, pady=20, command=run)
runbutton.grid(row=90,column=45)

global num1
num1 = StringVar()
numbmaxim = Entry(root, textvariable= num1 ,width=30).grid(row=25, column=45)
textfornumaxin = Label(root, text="Number of maximum address (ex: 350)")
textfornumaxin.grid(row=23, column=45)

global num2
num2 = StringVar()
numbofspot = Entry(root, textvariable= num2, width= 30).grid(row=29, column= 45)
textfornumspot = Label(root, text="Number of spot (ex: 50000000)")
textfornumspot.grid(row=27, column=45)

global num31
num31 = StringVar()
learn_rate_x = Entry(root, textvariable= num31, width= 30).grid(row= 33, column= 45)
textforrate = Label(root, text="Learning rate_x (recommendation: 0.35)")
textforrate.grid(row=31, column=45)

global num32
num32 = StringVar()
learn_rate_z = Entry(root, textvariable= num32, width= 30).grid(row= 37, column= 45)
textforrate = Label(root, text="Learning rate_z (recommendation: 0.35)")
textforrate.grid(row=35, column=45)

global num4
num4 = StringVar()
threshold_value = Entry(root, textvariable= num4, width= 30).grid(row= 41, column= 45)
textforthreshold = Label(root, text="BG threshold (example: 1500 (min intensity))")
textforthreshold.grid(row=39, column=45)

global num5
num5 = StringVar()
reblur_value = Entry(root, textvariable= num5, width= 30).grid(row= 45, column= 45)
textforreblur = Label(root, text="reblurring factor (recommendation: 0)")
textforreblur.grid(row=43, column=45)

#shoving it onto the screen
root.mainloop()



# In[ ]:




