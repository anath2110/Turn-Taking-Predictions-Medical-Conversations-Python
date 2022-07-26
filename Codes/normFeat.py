# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:57:36 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA 
Translated to Python and modified from the original in MATLAB,
zNormalizeBothSpeakers.m by Gerardo Cervantes.
Reference:
Ward et al., "TURN-TAKING PREDICTIONS ACROSS LANGUAGES AND GENRES USING AN LSTM
RECURRENT NEURAL NETWORK", IEEE Workshop on Spoken Language Technology (SLT),2018
"""
"""
This script handles Z-normalizing 3D multidimensional arrays, where the
features are in the first dimension, axis 0
Or of 2D numpy arrays where features are in axis 2.
"""

def normFeat(features, featsMean, featsStd, feature_index,shape):    
    if (shape=='3D' or shape=='3d'):
        featuresAtIndex = features[feature_index,:, :]
        featuresAtIndex = featuresAtIndex - featsMean[feature_index]
        featuresAtIndex = featuresAtIndex/featsStd[feature_index]
        features[feature_index,:,:] = featuresAtIndex    
        normalizedFeats = features
    elif (shape=='2D' or shape=='2d'):
        featuresAtIndex = features[:, feature_index]
        featuresAtIndex = featuresAtIndex - featsMean[feature_index]
        featuresAtIndex = featuresAtIndex/featsStd[feature_index]
        features[:,feature_index] = featuresAtIndex    
        normalizedFeats = features
        
    return normalizedFeats

    