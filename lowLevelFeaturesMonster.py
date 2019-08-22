# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:07:38 2019

@author: Anindita Nath
3M| M*Modal
Pittsburgh, PA

Translated to Python and modified from the original in MATLAB,
makeTrackMonsterSimplified.m by Gerardo Cervantes.

Reference:
Ward et al., "TURN-TAKING PREDICTIONS ACROSS LANGUAGES AND GENRES USING AN LSTM
RECURRENT NEURAL NETWORK", IEEE Workshop on Spoken Language Technology (SLT),2018
"""
"""
Simplified version of makeTrackMonster.py
Returns 6 prosodic features every 10ms frames of a single track/channel
of an audio signal :
                    Absolute pitch, Relative Pitch,Energy,
                    Cepstral Flux and  Speaking Frames
Runs without featurefile '.fss'

Input: trackspec (output of makeTrackSpecs.py)
       trackletter 'l' or 'r'
"""
import readwav_structobject as wavstruct
import numpy as np
import math
from computeLogenergy import computeLogEnergy
from lookOrComputePitchWav import lookOrComputePitchWav
from cepstralFlux import cepstralFlux
from speakingFrames import speakingFrames

def lowLevelFeaturesMonster(pathToSavePitchCache,trackspec, trackletter):
     msPerFrame = 10
     signal = wavstruct.SignalObj(trackspec.path)
     rate=signal.fs
     samplesPerFrame = msPerFrame * int(rate / 1000)#11 for OAK data
     if(signal.channels==2):
         if(trackletter=='l'):
             signalOneTrack = signal.data[:,0]         
         elif(trackletter=='r'):
             signalOneTrack = signal.data[:,1] 
     elif(signal.channels==1):
             signalOneTrack = signal.data
         
     praw, pCenters = lookOrComputePitchWav(pathToSavePitchCache,trackspec.directory,trackspec.filename, trackletter,signal)
     energy = computeLogEnergy(signalOneTrack, samplesPerFrame)
     cepstral = cepstralFlux(signalOneTrack, rate, energy)
     pitch=praw
     
     pitch[np.isnan(pitch)]=0.0 #Convert NaN values in pitch to 0
     relativePitch = pitch #Z-normalized after everything is concatenated
     
     speakingFramesFromPitch = np.zeros(len(pitch),)# energy is better measure of speaking frames
     speakingFramesFromPitch[(pitch!=0)]=1 

     speakingFramesFromEnergy=speakingFrames(energy)
    
     absolutePitch = np.log(pitch) #Convert inf values in pitch to 0 (Since log(0) returns -inf)
     absolutePitch[np.isinf(absolutePitch)] = 0 #np.isinf checks for both pos and neg infinity
     
     return relativePitch,absolutePitch,energy,cepstral,speakingFramesFromPitch,speakingFramesFromEnergy