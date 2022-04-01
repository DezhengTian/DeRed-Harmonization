import datetime
from genericpath import exists
import time
import os
import keras.backend as K
import nibabel as nib
import numpy as np
import pandas as pd
import cv2


def loadWeights(model,path,orientation):
    path = '{}{}/{}.hdf5'.format(path,orientation,model.name)
    model.load_weights(path)

def postProcessing(img):
    img = img+1
    img = img*125
    img = img.transpose(1,0,2)[::-1,:,:]
    return img

def preProcessing(img):
    img = img[::-1,:,:].transpose(1,0,2)
    img = img/125
    img = img-1
    return img

def loadImage(path,orientation,target_site,source_site):
    target_site_path = '{}{}/{}/'.format(path,target_site,orientation)
    source_site_path = '{}{}/{}/'.format(path,source_site,orientation)
    target_img_group,source_img_group = [],[]
    target_site_path_list = os.listdir(target_site_path)
    source_site_path_list = os.listdir(source_site_path)
    for file_path in target_site_path_list:
        img_path = target_site_path+file_path
        img = cv2.imread(img_path)
        target_img_group.append(preProcessing(img))
    for file_path in source_site_path_list:
        img_path = source_site_path+file_path
        img = cv2.imread(img_path)
        source_img_group.append(preProcessing(img))
    return np.array(target_img_group),np.array(source_img_group),target_site_path_list

def saveImage(data,path,file_list):
    if not exists(path):
        os.makedirs(path)
    for _ in range(len(file_list)):
        cv2.imwrite('{}/harmonized_{}'.format(path,file_list[_]),postProcessing(data[_]))









