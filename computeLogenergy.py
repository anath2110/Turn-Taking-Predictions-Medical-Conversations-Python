# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:16:15 2017

@author:Anindita Nath 
        University of Texas at ElPaso
"""
import numpy as np
from readtracks import readtracks

def computeLogEnergy(signal, samplesPerWindow):

#Returns a vector of the energy in each frame.
#A frame is, usually, 10 milliseconds worth of samples.
#Frames do not overlap.
#Thus the values returned in logEnergy are the energy in the frames
#centered at 5ms, 15ms, 25 ms ...
 
#A typical call will be:   en = computeLogEnergy(signal, 80);

#Note that using the integral image risks overflow, so we convert to double.
#For a 10minute file, at 8K rate, there are only 5 million samples,
#and max absolute sample value is about 20,000, and we square them,
#so the cumsum should always be under 10 to the 17th, so should be safe.

# convereted from the original code in Matlab by Nigel Ward, UTEP, November 2014
    signal=signal.astype(float)
    squaredSignal = np.multiply(signal,signal)
    #print(squaredSignal)
    integralImage = np.cumsum(squaredSignal)
    #print(integralImage)
    integralImage=np.insert(integralImage,0,0)
    #print(integralImage)
    integralImageByFrame = integralImage[0 : len(integralImage) :samplesPerWindow]
    #print(integralImageByFrame)
    perFrameEnergysub= np.subtract(integralImageByFrame[1:],integralImageByFrame[0:(len(integralImageByFrame)-1)])
    #print(perFrameEnergy) 
    perFrameEnergy = np.sqrt(perFrameEnergysub)
    #print(perFrameEnergy) 
    
    #replace zeros with a small positive value (namely 1) to prevent log(0)
    
    perFrameEnergy[perFrameEnergy==0]= 1
    #print(perFrameEnergy)             
    
    logEnergy = np.log(perFrameEnergy)
    #print(logEnergy)
    return logEnergy
    #test cases:
    #computeLogEnergy([1 2 3 1 2 3 4 5 6], 2)
    #computeLogEnergy([1 2 3 4 5 6 5 4 3], 2)

#logenergy=computeLogEnergy([1,2,3,4,5,6,5,4,3], 3)

#[s,c,r] = readtracks('f0a_01.au')# stereo
#e = computeLogEnergy(s[:,1], 80)
#np.set_printoptions(threshold=np.nan)
#print(e)
#plot(1:length(e), e, (1:length(s))/80, s(:,1)/2000)
