#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:14:23 2019

@author: njj
"""


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import numpy as np
from glob import glob
import math
import keras
import cv2
import imageio
from keras import layers
from keras import ops

from PIL import Image, ImageEnhance

from keras import backend as K
import random
import torch
from keras import losses
from numpy import random

import scipy.misc
from keras.regularizers import L2

import matplotlib.pyplot as plt
from aug_utils import random_augmentation
from keras.models import Model
#from keras.utils import multi_gpu_model
from keras.layers import Input,Embedding,Conv1D, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,concatenate,Multiply,add
from keras.layers import AveragePooling2D,Flatten,Concatenate,Conv2DTranspose,GlobalMaxPooling2D,GlobalAveragePooling2D,Reshape,Dense,multiply, Permute,dot
from keras.layers import GlobalAveragePooling1D,DepthwiseConv2D,Activation,BatchNormalization,SeparableConv2D
from keras.optimizers import Adam,SGD,Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau ,Callback

from keras import regularizers

from random import randint

batch_size = 16
input_shape = (128,128)
nepochs=100
batch_size = 16
lr=0.003
def dice(y_true,y_pred):
    smooth=1.
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    intersection = (y_true * y_pred).sum()
    dice = (2.*intersection + smooth)/(y_true.sum() + y_pred.sum() + smooth)
    
    return dice


def gelu(X):
 
    return 0.5*X*(1.0 + K.tf.tanh(np.sqrt(2 / np.pi)*(X + 0.044715*K.tf.pow(X, 3))))

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


def convBn(x,chanels,filters):
    x1 = DepthwiseConv2D(filters, padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(x)
    x1=BatchNormalization()(x1)
    x1 = add([x, x1])
    x2=Activation("relu")(x1)
   
    return x2


def convBn1(x,chanels,filters):
    
    x1 = Conv2D(chanels,filters, padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(x)
    x1=BatchNormalization()(x1)

    x2=Activation("relu")(x1)
  
    return x2

def convBn2(x,chanels,filters):
    x1 = Conv2D(chanels,filters, activation='relu',padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(x)
    #x1=BatchNormalization()(x1)
    x2=DepthwiseConv2D(3,dilation_rate=(1,1), padding = 'same',depthwise_initializer="glorot_uniform")(x1)
 
    return x2
def RFDB(x,chanels,filters):
    #x1=convBn1(x,chanels,1)
    x1 = Conv2D(chanels,1, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(x)
    x2 =convBn(x,chanels,filters)
    #x3 =convBn1(x2,chanels,1)
    x3 = Conv2D(chanels,1, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(x2)
    x4 =convBn(x2,chanels,filters)
    #x5= convBn1(x4,chanels,1)
    x5 = Conv2D(chanels,1, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(x4)
    x6 =convBn(x4,chanels,filters)
  
    x7 = Conv2D(chanels,3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(x6)
    
   # x7= convBn2(x7,chanels,3)
    
    x8=Concatenate()([x1,x3])
    x9=Concatenate()([x8,x5])
    x10=Concatenate()([x9,x7])
   # x11=convBn1(x10,chanels,3)
    
    x11 = Conv2D(chanels,3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(x10)
 
    x13 = add([x, x11])
    
   
    x14 = Conv2D(chanels,1, padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.0001))(x13)
    x15=BatchNormalization()(x14)
    x16=Activation("relu")(x15)
    #x16=gelu(x15)
   # x=LeakReLU(alpha=0.3)(x)
    return x16


def path(x,kernel,stride):
    patch_size=2
 
    shape=x.shape
    print(shape)
   # shape = K.int_shape(x)
    b,c,h,filters1=shape
  
    path=Patches(patch_size)(x)
  
    shape1=path.shape
    #shape1 = K.int_shape(path)
    c1,h1,f1=shape1
    print(shape1)
   
    path= Dense(filters1, activation='relu', kernel_initializer='he_normal',kernel_regularizer=regularizers.L2(0.0001))(path)
  
    shape2 = path.shape
  
    print(shape2)
    
    return path

def squeeze_excite_block(input, ratio=8):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
   
    shape =init.shape
    b,c,h,filters=shape
    se_shape = (1, 1, filters)
    #filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def specificity(y_true, y_pred):
        """Compute the confusion matrix for a set of predictions.
    
        Parameters
        ----------
        y_pred   : predicted values for a batch if samples (must be binary: 0 or 1)
        y_true   : correct values for the set of samples used (must be binary: 0 or 1)
    
        Returns
        -------
        out : the specificity
        """
        neg_y_true = 1 - y_true
        neg_y_pred = 1 - y_pred
        fp = (neg_y_true * y_pred).sum()
        tn = (neg_y_true * neg_y_pred).sum()
        
        specificity = tn / (tn + fp + K.epsilon())
        return specificity
    
def sensitivity(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns:
    Sensitivity score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fn = (neg_y_pred * y_true).sum()
    tp = (y_true * y_pred).sum()
    
    sensitivity = tp / (tp + fn+ K.epsilon())
    return sensitivity
# Specificity (true negative rate)


def PPV(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns:
    PPV score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = np.sum(neg_y_pred * y_true)
    tp = np.sum(y_true * y_pred)
    
    PPV = tp / (tp + fp)
    return PPV
def NPV(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns: NPV
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred

    tn = np.sum(neg_y_true * neg_y_pred)
    
    NPV = tn / np.sum(neg_y_pred)
    return NPV

def dice_coef_loss(y_true, y_pred):
     return 1-dice(y_true, y_pred)


def ASPP(x,chanels):
   shape=x.shape
 #  shape = K.int_shape(x)
   #shape = K.int_shape(x)
   b,c,h,filters=shape
  # shape=K.int_shape(x)
   b0=DepthwiseConv2D(3,dilation_rate=(1,1), padding = 'same')(x)
#   b0=Conv2D(chanels,(3,3),padding="same",use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(x)
   b0=BatchNormalization()(b0)
   b0=Activation("relu")(b0)
   #b0=squeeze_excite_block(b0)
   b1=DepthwiseConv2D(3,dilation_rate=(3,3), padding = 'same')(x)    
#   b1=SeparableConv2D(chanels,(3,3),dilation_rate=(3,3),padding="same",use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(x)
   b1=BatchNormalization()(b1)
   b1=Activation("relu")(b1)

   b2=DepthwiseConv2D(3,dilation_rate=(5,5), padding = 'same')(x)
   #=SeparableConv2D(chanels,(3,3),dilation_rate=(5,5),padding="same",use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(x)
   b2=BatchNormalization()(b2)
   b2=Activation("relu")(b2)
  
   shape2=b2.shape
   b,c,h,filters1=shape2
   x1=add([b0, b1])
  # x1=Concatenate()([b0,b1])
   x1=Conv2D(filters,(3,3),padding="same",use_bias=False,kernel_regularizer=regularizers.L2(0.0001))(x1)
   x1=BatchNormalization()(x1)
   x1=Activation("relu")(x1)

   shape1=x1.shape
   b,c,h,filters=shape1

   se_shape = (1, 1, filters)

   se = GlobalAveragePooling2D()(b0)
   se = Reshape(se_shape)(se)
   se = Dense(filters // 2, activation='relu', kernel_initializer='he_normal')(se)
   se = Dense(filters1, activation='sigmoid', kernel_initializer='he_normal')(se)

   se1=Conv2D(1, [1, 1], strides=[1, 1])(b1)
   se1 = Activation('sigmoid')(se1)
 
   
   x2 = multiply([se1,b0])
   x3=multiply([se,b1])
   
 #  x4=add([x2,x3])
   
   x4=Concatenate()([x2,x3])
  
   b4=Conv2D(filters,(3,3),padding="same",kernel_regularizer=regularizers.L2(0.0001))(x4)
   b4=BatchNormalization()(b4)
   b4=Activation("relu")(b4)
   
   se2=GlobalAveragePooling2D()(b2)
   se2=Reshape(se_shape)(se2)
   se2=Dense(filters//4, activation="relu",kernel_initializer='he_normal')(se2)
   se2=Dense(filters,activation="sigmoid",kernel_initializer="he_normal")(se2)
    # b4=BilinearUpsampling((out_shape,out_shape))(b4)
   x5=multiply([se2,b4])
   x=Concatenate()([x5,x])

   return x


def attention_block_2d(x, g, inter_channel):
    # theta_x(?,g_height,g_width,inter_channel)
    print(inter_channel)
    theta_x = Conv2D(inter_channel, 1, strides=[1, 1])(x)

    # phi_g(?,g_height,g_width,inter_channel)  
    print(inter_channel)
    phi_g = Conv2D(inter_channel,1, strides=[1, 1])(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)

    rate = Activation('sigmoid')(psi_f)

  

    att_x = multiply([x, rate])

    return att_x

    
def non_lock1(x1,x2,x3,x4,x5,ratio=8):

   # x1=DepthwiseConv2D(3,dilation_rate=(3,3), padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.0001))(x1)
    x1 = convBn2(x1,64,3)
    x2 = convBn2(x2,64,3)
    x3 = convBn2(x3,64,3)
    x4 = convBn2(x4,64,3)
    x5 = convBn2(x5,64,3)
    
    x_shape=x2.shape
   # x_shape=K.int_shape(x2)
    batchsize1,dim11,dim21,channels1=x_shape
    
    
    x_shape1=x3.shape
   # x_shape1=K.int_shape(x3)
    b,d1,d2,c=x_shape1
   # x2=BilinearUpsampling((2,2))(x2)
    se_shape = (1,1,c)
    x1=path(x1,2,2)
#    f= GlobalAveragePooling2D()(x2)
    f = Reshape((-1,channels1))(x2)

    size11=f.shape
    print(size11)
 
    q=x1

    p = dot([f, q], axes=1)
   # size111=K.int_shape(p)
   # print("1")
   # print(size111)
    p = Activation('softmax')(p)

    theta=p
 
    phi=Reshape((-1,channels1))(x3)
  
    f1=dot([theta,phi],axes=2)
    
    #f1=path(f1,2,2)
    
    f1=Reshape((-1,d1*d2,channels1))(f1)
   
  
    x4 = GlobalAveragePooling2D()(x4)
    x4 = Reshape(se_shape)(x4)
    x4 = Dense(c // 4, activation='relu', kernel_initializer='he_normal')(x4)
    x4 = Dense(c, activation='sigmoid', kernel_initializer='he_normal')(x4)



    f3 = multiply([x4, f1])
    
    
    f3=Reshape((d1,d2,c))(f3)
  
    se = GlobalAveragePooling2D()(f3)
    se = Reshape(se_shape)(se)
    se = Dense(channels1 // ratio, activation='relu', kernel_initializer='glorot_uniform', use_bias=False)(se)
    se = Dense(channels1, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=False)(se)
    
    x = multiply([se, x5])
    x=Conv2D(c*2,(3,3),padding="same",use_bias=False,kernel_regularizer=regularizers.L2(0.0001))(x)
   # x=Concatenate()([x,x5])
    return x 

def non_lock(x,ratio=8): 
    x_shape=K.int_shape(x)
    batchsize1,dim11,dim21,channels1=x_shape
    f = Reshape((-1,channels1))(x)
  
    q = Reshape((-1,channels1))(x)
  
    p = dot([f, q], axes=2)
  
    p = Activation('softmax')(p)
   
    g=Conv2D(1, (1, 1), padding='same', use_bias=False, kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.L2(0.0001))(x)
    g = Reshape((-1, channels1))(x)

    
    y = dot([p, g], axes=[2, 1])
    
   
    y = Reshape((dim11,dim21,channels1))(y)
    y = add([x, y])
    
    return y
def semodule(input, x,ratio=2):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)
    
   # x1=Conv2D(filters//8,(1,1),padding="same",use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(init)
    
    init2 = x
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters1 = init2._keras_shape[channel_axis]
   
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // 4, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.L2(0.0001))(se)
    se = Dense(filters1, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=regularizers.L2(0.0001))(se)

    x3 = multiply([x, se])
    
    x3=Conv2D(filters1,(3,3),padding="same",use_bias=False,kernel_regularizer=regularizers.L2(0.0001))(x3)
    return x3
def prehead(x,conv1,depth,features):
    
    for i in reversed(range(depth)):
        features = features 
      
        x = UpSampling2D(size=(2, 2),interpolation="bilinear")(x)
#        x = concatenate([skips[i], x], axis=3)
        x = DepthwiseConv2D(3,dilation_rate=(3,3), padding = 'same')(x)
#        x = Dropout(0.2)(x)
        x = DepthwiseConv2D(3,dilation_rate=(5,5), padding = 'same')(x)
    
    conv7=Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(x)
    #conv8=convBn1(conv7,16,3)
    conv8=Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(conv7)
    
   # conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)
    return conv8
def read_input(path):
    x = np.array(cv2.imread(path))/255.
    return x


def read_gt(path):
    x = np.array(Image.open(path))/255.
    return x[..., np.newaxis]


def random_crop(img, mask, crop_size=input_shape[0]):
    imgheight= img.shape[0]
    imgwidth = img.shape[1]
    i = randint(0, imgheight-crop_size)
    j = randint(0, imgwidth-crop_size)

    return img[i:(i+crop_size), j:(j+crop_size), :], mask[i:(i+crop_size), j:(j+crop_size)]


def gen(data, au=False):
    while True:
        repeat = 5
        index= random.choice(list(range(len(data))), batch_size//repeat)
        index = list(map(int, index))
       # print(index)
        list_images_base = [read_input(data[i][0]) for i in index]
      #  print("!!!1")
       # print(list_images_base)
        list_gt_base = [read_gt(data[i][1]) for i in index]

        list_images = []
        list_gt = []
        list_gt1_aug=[]
        list_gt2_aug=[]
        list_gt3_aug=[]

        for image, gt in zip(list_images_base, list_gt_base):

            #if au:
            for _ in range(repeat):
                image_, gt_ = random_crop(image.copy(), gt.copy())
                list_images.append(image_)
                list_gt.append(gt_)

        list_images_aug = []
        list_gt_aug = []

        for image, gt in zip(list_images, list_gt):
            #for _ in range(repeat):
            if au:
                image, gt = random_augmentation(image, gt)
            list_images_aug.append(image)
            list_gt_aug.append(gt)
            list_gt1_aug.append(gt)
            list_gt2_aug.append(gt)
            list_gt3_aug.append(gt)

        yield np.array(list_images_aug), np.array(list_gt_aug)
        
def step_decay(epoch):
    initial_lrate = lr
    total_epoch=nepochs
    drop=0.9
    lrate = initial_lrate * math.pow((1-((epoch-1)/total_epoch)),drop)
    return lrate


def newmyunet(patch_height,patch_width,n_ch,weight_decay=1e-4):
    
    #inputs = Input((None, None, 3))
    inputs = Input((patch_height,patch_width,n_ch))
    skips = []
    conv1 = Conv2D(32,3, activation='relu',padding="same",kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(inputs)
     
    conv1 = convBn2(conv1,32,3)
    skips.append(conv1)
  
    pool11 = MaxPooling2D(pool_size=(2, 2))(conv1)
   
    conv2=convBn1(pool11,32,3)
 
    conv2 = convBn1(conv2,64,3)
#
    xconv2=prehead(conv2,conv1,1,64)
 
    skips.append(conv2)
  # 
    pool21 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3=convBn1(pool21,64,3)
  
    conv3 = convBn1(conv3,128,3)
  
    xconv3=prehead(conv3,conv1,2,128)
    skips.append(conv3)
#    pool3=path(conv1,8,8)
#    pool3=convBn1(pool3,128,3)
    pool31 = MaxPooling2D(pool_size=(2, 2))(conv3)
 
    conv4= convBn1(pool31,128,3)
#   
    conv4= convBn1(conv4,128,3)
 
    xconv4 =prehead(conv4,conv1,3,128)
    skips.append(conv4)
    
    pool41 =MaxPooling2D(pool_size=(2,2))(conv4)
    
   # conv5=RFDB(pool4, 128, 3)
    conv5=convBn1(pool41, 128, 3)
   # conv5=add([conv5,pool4])
    
 
    conv5= convBn1(conv5,128,3)
#    conv5=BatchNormalization()(conv5)
    #x3=BilinearUpsampling((16,16))(conv5)
   # conv5=ASPP(conv5,128)
    x3=non_lock1(conv2,conv3,conv4,conv5,conv1)
#
    x11=convBn1(x3, 32, 3)
   # x11=BatchNormalization()(x11)
    
   # xc=convBn1(xc,32,3)
   # xc = Conv2D(32,3,activation = 'relu', padding = 'same',kernel_initializer = 'glorot_uniform',kernel_regularizer=regularizers.l2(0.0001))(xc)
    x11=ASPP(x11,32)
   
    x12=Concatenate()([xconv2, xconv3])
    x13=Concatenate()([x12, xconv4])
   
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(x13)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.L2(0.0001))(conv9)
     
     
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  
   
    model = Model(inputs = [inputs], outputs = [conv10])
  
    model.summary()
     #model.load_weights('unet.hdf5')

    model.compile(optimizer="adamw", loss=losses.binary_crossentropy,metrics = ['accuracy',specificity,sensitivity,dice])
       
    return  model


if __name__ == "__main__":
    
     weight_decay = 1e-4
    
     model_name = "baseunet"

     print("Model : %s"%model_name)

     train_data = list(zip(sorted(glob('E:\\maching learning\\data\\DRIVE\\training\\images\\*.tif')),
                           sorted(glob('E:\\maching learning\\data\\DRIVE\\training\\1st_manual\\*.gif'))))

     print(len(train_data))
  
     model = newmyunet(128,128,3)
  
     file_path = model_name + ".weights.h5"
   

     reduce_lr = LearningRateScheduler(step_decay)
   
     history = model.fit(gen(train_data, au=True), epochs=nepochs, verbose=2,
                          steps_per_epoch= 100*len(train_data)//batch_size,
                                    callbacks=[reduce_lr])
 	
     
     model.save_weights(file_path)