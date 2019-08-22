# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:57:36 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA   
Translated to Python and modified from the original in MATLAB, 
getFeaturesMeanStd.m by Gerardo Cervantes.

Reference:
Ward et al., "TURN-TAKING PREDICTIONS ACROSS LANGUAGES AND GENRES USING AN LSTM
RECURRENT NEURAL NETWORK", IEEE Workshop on Spoken Language Technology (SLT),2018
"""

"""
Find the mean and standard deviation of the
features.  
"""
import numpy as np


def computeStats(Features, shape):
    #print(trainFeatures.shape)

#     #for 3d features, where axis=0 are the features, 
#     #axis 1 rows and axis 2 columns
    if (shape=='3D' or shape=='3d'):
             samplemeans=np.mean(Features,axis=1)
             #print(samplemeans.shape)
             #samplesMeans = np.squeeze(means).shape
             #print(samplesMeans.shape)
             featuresMean = np.mean(samplemeans,axis=1)
             #print(featuresMean.shape)
             samplesStds = np.std(Features,axis=1)
             #print(samplesStds.shape)
             featuresStd = np.std(samplesStds,axis=1)
             #print(featuresStd.shape)

    elif (shape=='2D' or shape=='2d'):
       
        featuresMean = np.mean(Features,axis=0)
        #print(featuresMean.shape)
        
        featuresStd = np.std(Features,axis=0)
        #print(featuresStd.shape)
    return featuresMean,featuresStd