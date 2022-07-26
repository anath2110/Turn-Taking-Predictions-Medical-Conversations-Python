## -*- coding: utf-8 -*-
#"""
#Created on Mon Jul  1 16:52:04 2019
#
#@author: anindita.nath
#"""

#
import sys
import numpy as np
import math
import _pickle as pickle
import tensorflow as tf
import tensorflow # this sets KMP_BLOCKTIME and OMP_PROC_BIND
import os
del os.environ['OMP_PROC_BIND']
del os.environ['KMP_BLOCKTIME']



with tf.device('/CPU:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

import tensorflow as tf

n_cpus = 20

sess = tf.Session(config=tf.ConfigProto(
    device_count={ "CPU": n_cpus },
    inter_op_parallelism_threads=n_cpus,
    intra_op_parallelism_threads=1,
))

size = 100000

A = tf.ones([size, size], name="A")
B = tf.ones([size, size], name="B")
C = tf.ones([size, size], name="C")

with tf.device("/cpu:0"):
    x = tf.matmul(A, B)
with tf.device("/cpu:1"):
    y = tf.matmul(A, C)

sess.run([x, y])
#print(bool(int(sys.argv[1])))

#features=np.reshape(np.arange(30),(-1,5,2))
#print(features.shape)
#print(features)
##x = np.reshape(features, [-1, 2])
#x=np.transpose(features,[2,1,0])
#x=np.transpose(x,[1,0,2])
#print(x.shape)
#print(x)
#features2D= np.arange(30)
#print(features2D.shape)
#featuresPastCurrentFuture=np.zeros((features2D.shape[0]-20,21))
##CurrentFeatures
#featuresPastCurrentFuture[:,10]=features2D[10:-10]
#print(featuresPastCurrentFuture.shape)
#
#for row in range(featuresPastCurrentFuture.shape[0]):
#    #PastFeatures
#    #10frames to the past
#    featuresPastCurrentFuture[row,:10]=features2D[row:row+10].T
#    
#    #FutureFeatures
#    #10 frames to the future
#    featuresPastCurrentFuture[row,11:]=features2D[row:row+10].T
#   
#print(featuresPastCurrentFuture.shape)

# #read .pkl  files from cache
#with open('D:/FUNDataCodes/FeatsLabels/sampleOAK3Dialogs/featsLabels.pkl', 'rb') as f:
#     featslabels = pickle.load(f)
##each variable stored inpickle is numbered sequentially in featslabels
#print(len(featslabels))
#print(featslabels[0].shape)
#print(featslabels[1].shape)
#features2D= np.reshape(np.arange(180),(30,6))
#print('features2D shape')
#print(features2D.shape)
#print(features2D)
#print(features2D.T)
#features3D= np.reshape(features2D.T, (6,-1,10))
#print('features3D shape')
#print(features3D.shape)
#print(features3D)

#targets1D= np.arange(16760)
#print('targets1D shape')
#print(targets1D.shape)
#print(targets1D)
#targets2D= np.reshape(np.arange(83775),[5,-1])
#print('targets2D shape')
#print(targets2D.shape)
##print(targets2D)
#rem=(targets2D.shape[0] * targets2D.shape[1]) %(1200*5)
#print(((targets2D.shape[0] * targets2D.shape[1])-rem)/5)
#taketheseonly=int(((targets2D.shape[0] * targets2D.shape[1])-rem)/5)
#thisindex=targets2D.shape[1]-taketheseonly
#print(thisindex)
#div=((targets2D.shape[0] * targets2D.shape[1])-rem)/(1200*5)
#print(rem)
#print(div)
#targets2D=targets2D[:,:targets2D.shape[1]-thisindex]
#print(targets2D.shape)
#targets3D= np.reshape(targets2D, (5,-1,1200))
#print('targets3D shape')
#print(targets3D.shape)
#print(targets3D)
#trainFeatures=[]
#npa = np.asarray(trainFeatures, dtype=np.float64)
#npa = np.zeros((1,3,2))
#print(npa.shape)
#print(npa)
#features=np.reshape(np.arange(6),(1,-1,2))
#print(features.shape)
#trainFeatures = np.vstack((npa, features))
#print(trainFeatures.shape)
#targets=np.reshape(np.arange(360000),(600,600))
#windowSize=60
##print(targets)
#print(targets.shape)
##print(len(targets))
#targetMatrix =np.zeros((math.floor(len(targets)-windowSize), windowSize))
#print(targetMatrix.shape)
##
#for i in range(len(targets)-windowSize):
#        targetMatrix[i]= targets[i:windowSize+i,0]
#print(targetMatrix.shape)



#a = np.array([[1, 2]])
#b = np.array([[5, 6]])
#c=  np.array([[7, 8]])
#print(np.concatenate((a,b,c), axis=0))

## create a 2D array
#a = np.array([[1,2,3], [4,5,6], [1,2,3], [4,5,6],[1,2,3], [4,5,6],[1,2,3], [4,5,6]])
#
#print(a.shape) 
## shape of a = (8,3)
#
#b = np.reshape(a, (8, -1, 3)) 
## changing the shape, -1 means any number which is suitable
#
#print(b.shape) 
# size of b = (8,3,1)
##pitch=np.array([1, 3, np.NaN, 58])
###pitch = np.concatenate((pitch, np.nan),axis=0)
##print(pitch)
##pitch[np.isnan(pitch)]=0.0
##print(pitch)
##
##speakingFrames = np.zeros(len(pitch),)
##speakingFrames[(pitch!=0)]=1
##
##print(speakingFrames)
##
##
##
##absolutePitch = np.log(pitch)
##print(absolutePitch)
##absolutePitch[np.isinf(absolutePitch)]=0
##print(absolutePitch)
#
#
#
## Create a 3 dimensional ndarray
#
#nd_array = np.array([[[5,5,5,5],
#
#                     [6,6,6,6],
#
#                     [7,7,7,7]],
#
# 
#
#                    [[8,8,8,8],
#
#                     [9,9,9,9],
#
#                     [9,9,9,9]],
#
#                   
#
#                    [[10,10,10,10],
#
#                     [11,11,11,11],
#
#                     [12,12,12,12]]]
#
#                 )
#
#print("Input Array:")
#
#print(nd_array)
#
# 
#
#print("Shape of the array:")
#
#print(nd_array.shape)
#
# 
#
#print("Dimensions of the array:")
#
#print(nd_array.ndim)
#
# 
#
#print("Mean of a numpy.ndarray object - No axis specified:")
#
#print(nd_array.mean())
#
# 
#
#print("Mean of a numpy.ndarray object  - Along axis 0:")
#
#print(nd_array.mean(axis=0))
#
# 
#
#print("Mean of a numpy.ndarray object  - Along axis 1:")
#
#print(nd_array.mean(axis=1))
#
# 
#
#print("Mean of a numpy.ndarray object  - Along axis 2:")
#
#print(nd_array.mean(axis=2))
#
# 
#
#print("Variance of a numpy.ndarray object - No axis specified:")
#
#print(nd_array.var())
#
# 
#
#print("Variance of a numpy.ndarray object  - Along axis 0:")
#
#print(nd_array.var(axis=0))
#
# 
#
#print("Variance of a numpy.ndarray object  - Along axis 1:")
#
#print(nd_array.var(axis=1))
#
# 
#
#print("Variance of a numpy.ndarray object  - Along axis 2:")
#
#print(nd_array.var(axis=2))
#
# 
#
#print("Standard deviation of a numpy.ndarray object - No axis specified:")
#
#print(nd_array.std())
#
# 
#
#print("Standard deviation of a numpy.ndarray object - Along axis 0:")
#
#print(nd_array.std(axis=0))
#
# 
#
#print("Standard deviation of a numpy.ndarray object - Along axis 1:")
#
#print(nd_array.std(axis=1))
#
# 
#
#print("Standard deviation of a numpy.ndarray object - Along axis 2:")
#
#print(nd_array.std(axis=2))  
#
#
#x = np.array([[[0], [1], [2]]])
#x.shape
##(1, 3, 1)
#np.squeeze(x).shape
##(3,)
#np.squeeze(x, axis=0).shape
##(3, 1)