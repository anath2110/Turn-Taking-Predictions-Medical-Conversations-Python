# -*- coding: utf-8 -*-
from __future__ import division
"""
Created on Tue Aug  6 18:49:13 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA
"""
"""
Computes log Mel filter bank features from any given audio signal.
Returns numpy array of size #frames by #filters
"""
import numpy as np
from readtracks import readtracks
import readwav_structobject as wavstruct
import base
from scipy.signal import medfilt
import scipy
import os

def computeLFBE(pathToSaveFeatureCache,trackspec, trackletter):
    featureCacheDir = pathToSaveFeatureCache + 'LFBECachePython/'
    
    if not os.path.exists(featureCacheDir):
        os.makedirs(featureCacheDir)  
   
    #name of the feature file in .mat format
    featureStorageFileName = featureCacheDir +trackspec.filename + trackspec.side + '.mat' 
    
    
    signal = wavstruct.SignalObj(trackspec.path)
    rate=signal.fs    
    if(signal.channels==2):
         if(trackletter=='l'):
             signalOneTrack = signal.data[:,0]         
         elif(trackletter=='r'):
             signalOneTrack = signal.data[:,1] 
    elif(signal.channels==1):
             signalOneTrack = signal.data
    signal = np.array(signalOneTrack)
    
    #64 dimensions as mentioned in Amazon Paper
    LFBEfeatures=base.logfbank(signal,samplerate=rate,winlen=0.025,winstep=0.01,
          nfilt=64,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)
     
     
     #save features as .mat        
    scipy.io.savemat(featureStorageFileName, {'LFBEfeatures': LFBEfeatures})
    return LFBEfeatures