# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
Muthu Mookiah modified lines 115, created the functions imgResize and imgResizeOri on 2025/04/14.
"""

import numpy as np
import cv2
import os

def imgResize(image):
    # dim = (int(image.shape[1] *scale), int(image.shape[0] * scale))
    dim = (512, 512)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def imgResizeOri(image,dim):
    # dim = (int(image.shape[1]), int(image.shape[0]))
    resized = cv2.resize(image, dim, interpolation=cv2.NEAREST)
    return resized

def rgb2gray(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_test_patches(img, config,rl=True):

    test_img = []
    test_img.append(img)
    test_img=np.asarray(test_img)
    test_img_adjust=img_process(test_img,rl=rl)
    test_imgs=paint_border(test_img_adjust,config)
    test_img_patch=extract_patches(test_imgs,config)

    return test_img_patch,test_imgs.shape[1],test_imgs.shape[2],test_img_adjust

def paint_border(imgs,config):

    assert (len(imgs.shape) == 4)
    img_h = imgs.shape[1]  # height of the full image
    img_w = imgs.shape[2]  # width of the full image
    print('img height',img_h)
    print('img width',img_w)
    print(imgs.shape)
    leftover_h = (img_h - config.patch_height) % config.stride_height  # leftover on the h dim
    leftover_w = (img_w - config.patch_width) % config.stride_width  # leftover on the w dim
    print(leftover_h)
    print(leftover_w)
    full_imgs=None
    if (leftover_h == 0):  #change dimension of img_h
        tmp_imgs = np.zeros((imgs.shape[0],img_h+(config.stride_height-leftover_h),img_w,imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0],0:img_h,0:img_w,0:imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
    if (leftover_w == 0):   #change dimension of img_w
        tmp_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_w+(config.stride_width - leftover_w),full_imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0],0:imgs.shape[1],0:img_w,0:full_imgs.shape[3]] =imgs
        full_imgs = tmp_imgs
    print("new full images shape: \n" +str(full_imgs.shape))
    return full_imgs

def extract_patches(full_imgs, config):

    assert (len(full_imgs.shape)==4)  #4D arrays
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image

    assert ((img_h-config.patch_height)%config.stride_height==0 and (img_w-config.patch_width)%config.stride_width==0)
    N_patches_img = ((img_h-config.patch_height)//config.stride_height+1)*((img_w-config.patch_width)//config.stride_width+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print(N_patches_img)
    print(full_imgs.shape[0])
    print(N_patches_tot)
    patches = np.empty((N_patches_tot,config.patch_height,config.patch_width,full_imgs.shape[3]))# Original
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-config.patch_height)//config.stride_height+1):
            for w in range((img_w-config.patch_width)//config.stride_width+1):
                patch = full_imgs[i,h*config.stride_height:(h*config.stride_height)+config.patch_height,w*config.stride_width:(w*config.stride_width)+config.patch_width,:]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches

def pred_to_patches(pred,config):

    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    pred_images = np.empty((pred.shape[0],pred.shape[1],config.seg_num+1))  #(Npatches,height*width)# Original
    pred_images[:,:,0:config.seg_num+1]=pred[:,:,0:config.seg_num+1]
    pred_images = np.reshape(pred_images,(pred_images.shape[0],config.patch_height,config.patch_width,config.seg_num+1))
    return pred_images

def img_process(data,rl=False):

    assert(len(data.shape)==4)
    data=data.transpose(0, 3, 1,2)
    print('shape data',data.shape)
    if rl==False:
        # train_imgs=np.zeros(data.shape)
        if os.path.exists('trimgs.npy'):
            train_imgs = np.memmap('trimgs11.npy', dtype='float32', mode='w+', shape=data.shape)
        else:
            train_imgs = np.memmap('trimgs.npy', dtype='float32', mode='w+', shape=data.shape)
        for index in range(data.shape[1]):

            if os.path.exists('trimgsa.npy'):
                train_img = np.memmap(os.path.join('trimgsa11.npy'), dtype='float32', mode='w+',
                                      shape=(data.shape[0], 1, data.shape[2], data.shape[3]))
            else:
                train_img = np.memmap(os.path.join('trimgsa.npy'), dtype='float32', mode='w+',
                                      shape=(data.shape[0], 1, data.shape[2], data.shape[3]))
            train_img[:,0,:,:]=data[:,index,:,:]
            train_img = train_img/255.
            print('shape train img',train_img.shape)
            train_imgs[:,index,:,:]=train_img[:,0,:,:]


    else:
        train_imgs = np.zeros(data.shape)
        for index in range(data.shape[1]):
            print('testing preprocessing')
            train_img = np.zeros([data.shape[0], 1, data.shape[2], data.shape[3]])
            train_img[:, 0, :, :] = data[:, index, :, :]
            print('test preprocessing')
            train_imgs[:, index, :, :] = train_img[:, 0, :, :]/255.

    train_imgs=train_imgs.transpose(0, 2, 3, 1)
    return train_imgs


def img_processgt(data,rl=False):

    assert(len(data.shape)==4)

    print('shape gt data',data.shape)
    if rl==False:
        if os.path.exists('gtimgs.npy'):
            gt_imgs = np.memmap('gtimgs11.npy', dtype='float32', mode='w+', shape=data.shape)
        else:
            gt_imgs = np.memmap('gtimgs.npy', dtype='float32', mode='w+', shape=data.shape)
        for index in range(data.shape[1]):
            # train_img=np.zeros([data.shape[0],1,data.shape[2],data.shape[3]])
            if os.path.exists('gtimgsa.npy'):
                gt_img = np.memmap(os.path.join('gtimgsa11.npy'), dtype='float32', mode='w+',
                                      shape=(data.shape[0], 1, data.shape[2], data.shape[3]))
            else:
                gt_img = np.memmap(os.path.join('gtimgsa.npy'), dtype='float32', mode='w+',
                                      shape=(data.shape[0], 1, data.shape[2], data.shape[3]))
            gt_img[:,0,:,:]=data[:,index,:,:]

            gt_img = gt_img / 255.# Original

            gt_imgs[:,index,:,:]=gt_img[:,0,:,:]

    else:
        gt_imgs = np.zeros(data.shape)
        for index in range(data.shape[1]):
            gt_img = np.zeros([data.shape[0], 1, data.shape[2], data.shape[3]])
            gt_img[:, 0, :, :] = data[:, index, :, :]
            gt_imgs[:, index, :, :] = gt_img[:, 0, :, :]/255.

    return gt_imgs




def recompone_overlap(preds,config,img_h,img_w):

    assert (len(preds.shape)==4)  #4D arrays

    patch_h = config.patch_height
    patch_w = config.patch_width
    N_patches_h = (img_h-patch_h)//config.stride_height+1
    N_patches_w = (img_w-patch_w)//config.stride_width+1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " +str(N_patches_h))
    print("N_patches_w: " +str(N_patches_w))
    print("N_patches_img: " +str(N_patches_img))
    #assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//config.stride_height+1):
            for w in range((img_w-patch_w)//config.stride_width+1):
                full_prob[i,h*config.stride_height:(h*config.stride_height)+patch_h,w*config.stride_width:(w*config.stride_width)+patch_w,:]+=preds[k]
                full_sum[i,h*config.stride_height:(h*config.stride_height)+patch_h,w*config.stride_width:(w*config.stride_width)+patch_w,:]+=1
                k+=1
    print(k,preds.shape[0])
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    print('using avg')
    return final_avg



#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================


def imgs_zerone(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    if os.path.exists('zo.npy'):
        imgs_zo = np.memmap(os.path.join('zo11.npy'), dtype='float64', mode='w+', shape=imgs.shape)
    else:
        imgs_zo = np.memmap(os.path.join('zo.npy'), dtype='float64', mode='w+', shape=imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_zo[i,0] = (np.array(imgs[i,0], dtype = np.uint8))/255.
    return imgs_zo

#===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1


    if os.path.exists('norimg.npy'):
        imgs_normalized = np.memmap(os.path.join('norimg11.npy'), dtype='float64', mode='w+', shape=imgs.shape)
    else:
        imgs_normalized = np.memmap(os.path.join('norimg.npy'), dtype='float64', mode='w+', shape=imgs.shape)
    imgs_std = np.std(imgs) # Original
    imgs_mean = np.mean(imgs) # Original
    imgs_normalized = (imgs-imgs_mean)/imgs_std # Original

    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # imgs_equalized = np.empty(imgs.shape)
    if os.path.exists('equimg.npy'):
        imgs_equalized = np.memmap(os.path.join('equimg11.npy'), dtype='float64', mode='w+', shape=imgs.shape)
    else:
        imgs_equalized = np.memmap(os.path.join('equimg.npy'), dtype='float64', mode='w+', shape=imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values

    invGamma = 1.2 / gamma # (1.2/1.5 & 1.0/1.2) resulted best sensitivity
    print('gamma value',invGamma)

    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    # new_imgs = np.empty(imgs.shape)
    if os.path.exists('gamimg.npy'):
        new_imgs = np.memmap(os.path.join('gamimg11.npy'), dtype='float64', mode='w+', shape=imgs.shape)
    else:
        new_imgs = np.memmap(os.path.join('gamimg.npy'), dtype='float64', mode='w+', shape=imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs


