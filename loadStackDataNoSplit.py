# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:31:40 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA       

"""



import scipy.io as sio
import numpy as np
import os
import random

"""
Function reads the data from external data structures (.mat files).
Returns the stacked set of features and labels 
after changing the labels 
and/or deleting the unknown label and features if the corresponding flags are on.
""" 
def loadStack(pathToStoredDataDir,fileNames,startIndex,nFiles,nDims,
              changeLabels,deleteUnknows): 
  
  
  features=np.zeros((1,nDims))
  labels=np.zeros((1,1)) 
 
  
  countAll=len(fileNames)#Total number of files in the dataset 
  stopIndex=nFiles+startIndex
  
  if stopIndex>countAll:
      nFiles=stopIndex-countAll
      stopIndex=nFiles+startIndex
      
   
  if startIndex<countAll:
         
             for index in range(startIndex,stopIndex):
                 try:

                      matfile = sio.loadmat(pathToStoredDataDir + fileNames[index])                  
                      feats = matfile['features']
                      lbls = matfile['labels'] # shape (1,nsamples)

                      features=np.vstack((features,feats))
                      #transpose labels array
                      #since rows columns get interchnaged for 1D arrays in .mat files
                      labels=np.vstack((labels,lbls.T))#shape(nSamples,1)

                 except Exception:
                      print(" Out of Bounds in training data")
                      continue
             features = np.delete(features, (0), axis=0)
             labels=np.delete(labels, (0), axis=0)
         
  
  """Change Labels 0 to -1, 1 to 0, and 2 to 1 """
  if changeLabels==True:
            labels[labels==0]=-1
            labels[labels==1]=0
            labels[labels==2]=1

       
            
  """ Delete the unkonwn labels(-1)and corresponding features
      Remember to change labels before deleting"""
  if deleteUnknows==True:
      
          features=features[(labels > -1).all(axis=1),:]
          labels=labels[(labels > -1).all(axis=1)]
          
  
  return features, labels

"""
Function reads the names of files from external data structures (.mat files), 
randomly shufflesthe indices, loads and stacks them,
writes the stacked matrices of features and labels to disk as .mat files
after changing the labels 
and/or deleting the unknown label and features if the corresponding flags are on.
Returns the path to this storage file.
"""
def loadShuffleSave(pathToStoredDataDir,typeOfModel,fileNames,nthShuffleBatch,
                    shuffleSize,nDims,typeOfDataSet,changeLabels,deleteUnknows):
    
    
    features=np.zeros((1,nDims))
    labels=np.zeros((1,1)) 
    countAll=len(fileNames)
 
    shuffledIndices= random.sample(range(countAll),shuffleSize)

     
    for idx in shuffledIndices:

        try:

              matfile = sio.loadmat(pathToStoredDataDir + fileNames[idx])                  
              feats = matfile['features']
              lbls = matfile['labels'] # shape (1,nsamples)

              features=np.vstack((features,feats))
              #transpose labels array
              #since rows columns get interchnaged for 1D arrays in .mat files
              labels=np.vstack((labels,lbls.T))#shape(nSamples,1)

        except Exception:
              print(" Out of Bounds in training data")
              continue
          
    """ Deleting the first additional row of zeros"""
    features = np.delete(features, (0), axis=0)
    labels=np.delete(labels, (0), axis=0)  
    
  
    """Change Labels 0 to -1, 1 to 0, and 2 to 1 """
    if changeLabels==True:
            labels[labels==0]=-1
            labels[labels==1]=0
            labels[labels==2]=1


       
            
    """ Delete the unkonwn labels(-1)and corresponding features
      Remember to change labels before deleting"""
    if deleteUnknows==True:
      
          features=features[(labels > -1).all(axis=1),:]
          labels=labels[(labels > -1).all(axis=1)] 
        
    """Store the shuffled data in disk"""
    saveShuffledSetDir = pathToStoredDataDir + '_' + typeOfModel +'_'+\
     str(typeOfDataSet) + 'ShuffledSet/'
    if not os.path.exists(saveShuffledSetDir):
        os.makedirs(saveShuffledSetDir)  
    saveShuffledSetName= saveShuffledSetDir + \
    'Batch' + str(nthShuffleBatch) +'.mat'
       
      
    sio.savemat(saveShuffledSetName, {'features':features,'labels': labels}, do_compression=True)

 
    return saveShuffledSetName


"""
Function loads and returns the features and lables arrays stored in the input storage file.

"""
def loadFromShuffledStorage(pathToShuffledStoredData):
    
    matfile = sio.loadmat(pathToShuffledStoredData)                  
    features = matfile['features']
    labels = matfile['labels'] # shape(nsamples,1)
    

#            
    return features,labels