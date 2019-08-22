# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:38:18 2017

@author: Anindita Nath
University of Texas at ElPaso
"""
import numpy as np
def  windowize(frameFeatures, msPerWindow):

#inputs:
#frameFeatures: features over every 10 millisecond frame,
#centered at 5ms, 15ms etc. 
#A row vector.
#msPerWindow: duration of window over which to compute windowed values
#output:
#summed values over windows of the designated size, 
#centered at 10ms, 20ms, etc.
#(the centering is off, at 15ms, etc, if msPerWindow is 30ms, 50ms etc)
#but we're not doing syllable-level prosody, so it doesn't matter.
#values are zero if either end of the window would go outside  
#what we have data for. 

# converted from code by Nigel Ward, UTEP, Feb 2015

    integralImage =np.insert(np.cumsum(frameFeatures),0,0)    
    framesPerWindow = int(msPerWindow / 10)
    windowSum = np.subtract(integralImage[(framesPerWindow):],integralImage[:(len(integralImage)-framesPerWindow)])

    
    #align so first value is for window centered at 10 ms 
    #(or 15ms if, an odd number of frames)
    headFramesToPad = int(np.floor(framesPerWindow / 2)) - 1  
    tailFramesToPad = int(np.ceil(framesPerWindow / 2)) - 1  
    tailFramesToPad = int(np.ceil(framesPerWindow / 2))
    windowValues = np.concatenate((np.zeros(headFramesToPad,),windowSum,np.zeros(tailFramesToPad,)),axis=0)
    
    return windowValues
#test cases:
#windowValues=windowize(np.array([0,1,1,1,2,3,3,3,1,1,1,2]), 20)
##windowize([0 1 1 1 2 3 3 3 1 1 1 2], 30])
#np.set_printoptions(threshold=np.nan)
#print(windowValues.shape)



