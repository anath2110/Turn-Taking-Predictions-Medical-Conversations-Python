# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:31:40 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA  
"""
"""
This is the top-level function.

Returns features and labels for the given audio.
No split of train or test data.
Returns only 1 set of features and labels for each auidio.
Also, writes the same to disk in a .mat file.
"""
import numpy as np
import os



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
    #concatenating features side by side
    features = np.vstack((relativePitch,absolutePitch,energy,cepstral,
                               speakingFramesFromPitch,speakingFramesFromEnergy))

    #need to transpose because after vstack shape is (6,nsamples)
    features=features.T
    
    labelFilePath = labelsDirPath + '/' + audio[:len(audio)-len('.wav')] + '.stm'
    print('Reading labels from '+ labelFilePath)
    labels = makeLabels(audioDirPath + audio,labelFilePath)

    return features,labels




def saveFeatsLabels(pathtosave,audio,features,labels):
    filetosave=pathtosave + '6dims/' + audio + '_10ms2DFts1DLbls'
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
         
"""Sample Run"""    
# make the split=0 if you want all the audios to be for any 1 feature and label set 


#Features, Labels=\
#featuresAndLabels('D:/FUNDataCodes/FeatsLabels/sampleOAK3Dialogs/',
#                  'D:/FUNDataCodes/FeatsLabels/sampleOAK3Dialogs/1FutureFrame/',
#                  '100002.wav',
#                  'D:/FUNDataCodes/Audios/sampleAudios/', 
#                  'D:/FUNDataCodes/NewTurnStm/sampleStms/',0)

