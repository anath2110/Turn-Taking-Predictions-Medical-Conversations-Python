# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:25:44 2017

@author:Anindita Nath 
        University of Texas at ElPaso
"""

import os.path 
import readau
import readwav
import warnings
import numpy as np
def readtracks(filename):
# Given a stereo/audio audio filename 
# reads the file and returns two signals 



     [filepath, basename]= os.path.split(filename)
     [audioname,audioext]=os.path.splitext(basename)
     if audioext==".au":  
        print('in readtracks')
        print(filename)
        [rate,signals,channels] = readau.readau(filename)
        #print('channels in readtracks')
        #print(channels)
     elif audioext==".wav":
          print('in readtracks')
          print(filename)
          print(channels)
          [rate,signals,channels,duration] = readwav.readwav(filename) 
          
     else:
         warnings.warn("unexpected file extension")
         print(audioext)   


    
     
     if channels != 2:
         warnings.warn ("not a stereo file")
         
     if (rate!=8000 and rate!=16000):
         #higher rates seem to confuse the pitch tracker,
        print('sorry: sampling rate must be 8000 or 16000, not %d\n', rate)
     return signals,channels,rate

#testcases
#signals, channels,rate=readtracks('2ndWeekendNewscastJuly292012.au') #mono
#signals, channels,rate=readtracks('f0a_01.au') #stereo
#np.set_printoptions(threshold=np.nan)
#print(signals,channels,rate)