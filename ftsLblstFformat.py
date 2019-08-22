# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:01:48 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA   
"""
"""
Function converts the numpy feature arrays to the required tensor-flow 3D shape
(nSamples,timesteps,nFeaturedimensions).
And the corresponding label arrays to the shape (nSamples,timesteps,nClasses).
"""
import numpy as np
from reshapeFtsLbls import reshapeFtsLbls

def tfFormat(features,labels,timesteps,nClasses): 
    nFeaturedimensions=features.shape[1]
    """ Initially, features is a 2D array (nSamples,nFeaturedimensions).
    
    In Python, 3D array, axis 0 is axis 3 in Matlab, axis 1 rows, axis 2 cols.
    Also, each col in 2D features of shape (nSamples,nFeaturedimensions) 
    represents a different feature vector derived from monster.
    Hence, rows and cols transposed before they can be reshaped to 3D"""
    features=features.T # now it changes to (nFeaturedimensions,nSamples)
    
    features3D=reshapeFtsLbls(features,timesteps,nFeaturedimensions)     
      
    #print(features3D.shape)
    #Now, get the tensor flow format (nSamples,timesteps,nFeaturedimensions)
    features3DTf=np.transpose(np.transpose(features3D,[2,1,0]),[1,0,2])
   
    """Repeat the above steps but now with the label array
    which is of shape(nSamples,1)"""
    
    reshapedLabels=reshapeFtsLbls(labels,timesteps,nClasses) 
    #print(reshapedLabels.shape)
    #Now, get the tensor flow format (nSamples,timesteps,nFeaturedimensions)
    labelsTf=np.transpose(np.transpose(reshapedLabels,[2,1,0]),[1,0,2])
    
 
    return features3DTf,labelsTf