# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:59:08 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA 
"""
"""
Original top-level function modified to extract prosody features for each 
current frame(10ms) and corresponding 10 frames from the past 
and 10 frames to the future.
Returns the features and labels array corresponding to each dialog audio.
Also, writes the same to disk in a .mat file.
"""
import numpy as np
import os
from getDialogsList import getDialogsList
#from filesWithExtension import filesWithExtension
from makeTrackSpecs import makeTrackSpecs
from lowLevelFeaturesMonster import lowLevelFeaturesMonster
from makeLabels import makeLabels
from creating3DFeatsLabels import creating3DFeatsLabels
from computeStats import computeStats
from normFeat import normFeat
import scipy.io as sio
import _pickle as pickle #cPickle version in Python3.x 
                        

def featuresAndLabels(pathToSavePitchCache,pathToSaveFeatsLabels,
                      audio,audioDirPath, labelsDirPath,split):
   
 
     #used with monos 
    [features, targets] = \
     featuresAndLabelsPerTrack(pathToSavePitchCache,audio, audioDirPath, labelsDirPath,'l')      

    
  
    """ Sometimes audio features and labels are a few ms off, 
   following function makes the features and labels equal sized. 
   The longer array is truncated to match the size of the shorter one."""

  
    minSize=min((targets.shape[0],features.shape[0]))
    targets=targets[:minSize]
    features=features[:minSize,:]
       

    
    """Following for z-normalizing relative pitch"""
    #Computes mean and standard deviation from train features, 
    #the same will be used for test features
  
    [featsMean, featsStd] = computeStats(features,'2D') #shape can be 3D or 2D
    
    #Z-normalize relative pitch at feature_index 0
    #1st index, i.e. 0th in Python is where relative pitch is stored
    feature_index = 0
    
    features = normFeat(features, featsMean, featsStd, feature_index,'2D')

    
    saveFeatsLabels(pathToSaveFeatsLabels,audio,features, targets)
    return features, targets




"""
Following function returns features and labels for every 10 ms frames
for any given single track of an audio (only track if it is a mono)
Inputs:
        audioFile is the path to the audio(including its full name,
                                           base name with extension)
        audioDirPath is the path to the directory containing the audios
        labelsDirPath is the path to the directory containing the labels
        (the stm files)
        trackSide is either 'l' or 'r' i.e. left or right track
"""
def featuresAndLabelsPerTrack(pathToSavePitchCache,audio, audioDirPath, labelsDirPath, trackSide):    
            
    #Compute features for given track
    trackspec = makeTrackSpecs(trackSide, audio, audioDirPath + '/')
    
    relativePitch,absolutePitch,energy,cepstral,speakingFramesFromPitch,speakingFramesFromEnergy = \
    lowLevelFeaturesMonster(pathToSavePitchCache,trackspec, trackSide)

    
    #sometimes, the feature arrays vary in length, so cut the longer ones to 
    #match the shortest
    minSizeArray=min(len(relativePitch),len(absolutePitch),len(energy),
                     len(cepstral),len(speakingFramesFromPitch),
                     len(speakingFramesFromEnergy))
    relativePitch=relativePitch[:minSizeArray]
    absolutePitch=absolutePitch[:minSizeArray]
    energy=energy[:minSizeArray]
    cepstral=cepstral[:minSizeArray]
    speakingFramesFromPitch=speakingFramesFromPitch[:minSizeArray]
    speakingFramesFromEnergy=speakingFramesFromEnergy[:minSizeArray]
    

    
    #concatenate past, present and future frames
    relativePitch=featuresPastPresentFuture(relativePitch)
    absolutePitch=featuresPastPresentFuture(absolutePitch)
    energy=featuresPastPresentFuture(energy)
    cepstral=featuresPastPresentFuture(cepstral)
    speakingFramesFromPitch=featuresPastPresentFuture(speakingFramesFromPitch)
    speakingFramesFromEnergy=featuresPastPresentFuture(speakingFramesFromEnergy)
    

   
    #concatenating features side by side
    features = np.hstack((relativePitch,absolutePitch,energy,cepstral,
                               speakingFramesFromPitch,speakingFramesFromEnergy))
 
    #If labelsDir is empty, then create speaking labels from features, 
    #'speaking frames'
    #else the labels generated from stm files
    if not os.listdir(labelsDirPath): # checking if labelsDirectory is empty
        print('Creating labels from audio ')
        labels = speakingFramesFromEnergy
    else:
        labelFilePath = labelsDirPath + '/' + audio[:len(audio)-len('.wav')] + '.stm'
        print('Reading labels from '+ labelFilePath)
        labels = makeLabels(audioDirPath + audio,labelFilePath)
        
    return features,labels

""" Following function returns a 2d array(nsamples, nfeatures)
where nfeatures is 3.
The 3 colns. contain features from # past frames, 
present frame and #future frames.
Input: The feature array of shape(nsamples,)
"""
def featuresPastPresentFuture(featureArray):
    #since we take 10 frames to past and 10frames to future    #each frame being 50ms duration
   
    featuresPastCurrentFuture=np.zeros((featureArray.shape[0]-20,21))
    #CurrentFeatures
    featuresPastCurrentFuture[:,10]=featureArray[10:-10]
    #print(featuresPastCurrentFuture.shape)
    
    for row in range(featuresPastCurrentFuture.shape[0]):
        #PastFeatures
        #10frames to the past
        featuresPastCurrentFuture[row,:10]=featureArray[row:row+10].T
        
        #FutureFeatures
        #10 frames to the future
        featuresPastCurrentFuture[row,11:]=featureArray[row:row+10].T
   
   #print(featuresPastCurrentFuture.shape)
    
    
    return featuresPastCurrentFuture

def saveFeatsLabels(pathtosave,audio,features,labels):
    filetosave=pathtosave + 'PastFuture126dims/' + audio + '_10ms2DFts1DLbls'
    """Remember in Matlab and Python 3D, axis 3 and axis 0 are interchanged.
    While loading and reading these matfiles in Matlab, do permute(3Darrayname,[2,3,1]."""
    sio.savemat(filetosave + '.mat', {
                                    'features':features,
                                    'labels':labels})
    """
   saving /pkl per file is avoided as it takes more space than .mat, 
   also not needed since features and labels are 2Ds and !D respectively.
   """
#    """ save as pickle, Python serializor, stores variable s between sessions"""
#    with open(filetosave + '.pkl', 'wb') as outfile:         
#          pickle.dump([features, labels],
#                      outfile, protocol=3)#3 is equivalent for Highest Protocol
         
#..............................................................................#

""" Sample Run """
# make the split=0 if you want all the audios to be for any 1 feature and label set 

##
#Features, Labels=\
#featuresAndLabels('D:/FUNDataCodes/FeatsLabels/sampleOAK3Dialogs/',
#                  'D:/FUNDataCodes/FeatsLabels/sampleOAK3Dialogs/1FutureFrame/',
#                  '100039.wav',
#                  'D:/FUNDataCodes/Audios/sampleAudios/', 
#                  'D:/FUNDataCodes/NewTurnStm/sampleStms/',0)
