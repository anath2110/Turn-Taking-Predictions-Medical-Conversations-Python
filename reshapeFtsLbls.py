# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:00:35 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA
"""
"""
Function returns reshaped arrays, 
shape required before converting to tensor-flow format.
"""
import numpy as np

def reshapeFtsLbls(array,timesteps,nCols):
    
    #sometimes, the #elements in the array donot allow us to reshape to 
    #the required dimensions, hence, following assures size compatibility 
   
   
 
   if(nCols==1):      
        rem=array.shape[0]%timesteps        
        array=array[:array.shape[0]-rem,:]
        reshapedArray= np.reshape(array, (nCols,-1,timesteps))
   elif(nCols>1):
        rem=(array.shape[0] * array.shape[1]) %(timesteps*nCols)
        shapeofaxis1=int(((array.shape[0] * array.shape[1])-rem)/nCols)
        takeUptoindex=array.shape[1]-shapeofaxis1
        array=array[:,:array.shape[1]-takeUptoindex]    
        # reshape to (nFeaturedimensions,nSamples, timesteps)
        reshapedArray = np.reshape(array,(nCols,-1, timesteps))
    
    
   return reshapedArray