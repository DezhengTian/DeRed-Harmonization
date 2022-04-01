import time
import os
# import Loss
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from keras.layers import Input
# from keras.models import Model
import Model
import Util



class DeRed():
    def __init__(self,orientation):
        self.orientation = orientation
        self.filter = 32
        self.data_path = '../Data/' 
        self.weights_path = '../Weights/ATV - HUH/'
        self.target_site_name = 'ATV'
        self.source_site_name = 'HUH'  
        self.shape = self.getOrientation(orientation)
        self.site_factor_encoder_source = Model.getSiteFactorEncoder(self.shape,self.filter,self.source_site_name)
        self.bran_factor_encoder_source = Model.getBranFactorEncoder(self.shape,self.filter,self.source_site_name)
        self.site_factor_encoder_target = Model.getSiteFactorEncoder(self.shape,self.filter,self.target_site_name)
        self.bran_factor_encoder_target = Model.getBranFactorEncoder(self.shape,self.filter,self.target_site_name)
        self.decoder_source = Model.getDecoder(self.shape,self.filter,self.source_site_name)
        self.decoder_target = Model.getDecoder(self.shape,self.filter,self.target_site_name)
        
    def getOrientation(self,orientation='sagittal'):
        if orientation=='transverse': 
            return (176, 208, 3)
        elif orientation=='coronal': 
            return (176, 176, 3)
        elif orientation=='sagittal': 
            return (208, 176, 3)
        else :
            raise Exception('Illegal declaration: \'{}\','.format(orientation)+
            ' Only [\'transverse\', \'coronal\', \'sagittal\'] allowed.')      

    def loadWeights(self):
        Util.loadWeights(self.site_factor_encoder_source,self.weights_path,self.orientation)
        Util.loadWeights(self.bran_factor_encoder_source,self.weights_path,self.orientation)
        Util.loadWeights(self.site_factor_encoder_target,self.weights_path,self.orientation)
        Util.loadWeights(self.bran_factor_encoder_target,self.weights_path,self.orientation)
        Util.loadWeights(self.decoder_source,self.weights_path,self.orientation)
        Util.loadWeights(self.decoder_target,self.weights_path,self.orientation)
        
    def predict(self):
        self.loadWeights()
        target_data,source_data,target_file_list = Util.loadImage(self.data_path,self.orientation,self.target_site_name,self.source_site_name)
        s1,s2,s3 = self.site_factor_encoder_target(target_data)
        b1,b2,b3,b4 = self.bran_factor_encoder_source(source_data)
        s1 = tf.reduce_mean(s1,axis=0,keepdims=True)+s1*0
        s2 = tf.reduce_mean(s2,axis=0,keepdims=True)+s2*0
        s3 = tf.reduce_mean(s3,axis=0,keepdims=True)+s3*0
        predicted_data = self.decoder_target.predict([s1,s2,s3,b1,b2,b3,b4])
        Util.saveImage(predicted_data,'../Results/{}_to_{}/{}/'.format(self.source_site_name,self.target_site_name,self.orientation),target_file_list)

        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

