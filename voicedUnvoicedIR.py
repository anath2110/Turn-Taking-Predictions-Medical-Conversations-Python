# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:38:18 2017

@author: Anindita Nath
University of Texas at ElPaso
"""
from speakingFrames import speakingFrames
import numpy as np
def voicedUnvoicedIR(logEnergy, pitch, msPerWindow):
    
  #compute voiced-unvoiced intensity ratio
  # converted from original Nigel Ward, UTEP, February 2017; rescaled May 2017

  #for windows which lack unvoiced frames or lack voiced frames,
  #return the average value, since we only care about departures
  #from the norm
  #there really should be two features (like the two features for pitch height):
  #- one a measure of the evidence for high vvir,
  #- one a measure of the evidence for low vvir
  #with no evidence for either in cases where there is no speech
  #However for economy I use just one feature.
  #It's normalized so 0 means no evidence, positive is high, and negative is low.
  #having a default of 0 is convenient since makeTrackMonster does padding with 0


  if (len(logEnergy) == len(pitch) + 1):   
      pitch=np.insert(pitch,0,np.nan)

  isSpeech = speakingFrames(logEnergy)
  
  isSpeech[np.where(isSpeech==1)]=True
  isSpeech[np.where(isSpeech==0)]=False        
      
  
  voicedSpeechVec = (~np.isnan(pitch) & isSpeech)
  unvoicedSpeechVec = (np.isnan(pitch) & isSpeech)
  
  nonVoicedEnergiesZeroed = np.multiply(voicedSpeechVec,logEnergy)
  nonUnvoicedEnergiesZeroed = np.multiply(unvoicedSpeechVec,logEnergy)

  vFrameCumSum = np.insert(np.cumsum(nonVoicedEnergiesZeroed),0,0)
  uFrameCumSum = np.insert(np.cumsum(nonUnvoicedEnergiesZeroed),0,0)

  vFrameCumCount = np.insert(np.cumsum(voicedSpeechVec),0,0)
  uFrameCumCount = np.insert(np.cumsum(unvoicedSpeechVec),0,0)

  framesPerWindow = int(msPerWindow / 10)
  framesPerHalfWindow = int(framesPerWindow / 2)
  
  vFrameWinSum=np.zeros(len(vFrameCumSum))
  uFrameWinSum=np.zeros(len(uFrameCumSum))
  vFrameCountSum=np.zeros(len(vFrameCumCount))
  uFrameCountSum=np.zeros(len(uFrameCumCount))

  for i  in range(len(pitch)):
    wStart =  i - framesPerHalfWindow
    wEnd = i + framesPerHalfWindow 
    #prevent out-of-bounds
    if(wStart < 1):
      wStart = 1
   
    if(wEnd > len(pitch)):
      wEnd = len(pitch)
 
    vFrameWinSum[i] = vFrameCumSum[wEnd] - vFrameCumSum[wStart]
    uFrameWinSum[i] = uFrameCumSum[wEnd] - uFrameCumSum[wStart]
    vFrameCountSum[i] = vFrameCumCount[wEnd] - vFrameCumCount[wStart]
    uFrameCountSum[i] = uFrameCumCount[wEnd] - uFrameCumCount[wStart]


  avgVoicedIntensity = np.divide(vFrameWinSum,vFrameCountSum)
  avgUnvoicedIntensity = np.divide(uFrameWinSum,uFrameCountSum)

  ratio = np.divide(avgVoicedIntensity,avgUnvoicedIntensity)
  #exclude zeros, NaNs, and Infs
 
  averageOfValid =np.mean(ratio[~np.isinf(ratio) & (ratio>0)])
  ratio = np.subtract(ratio,averageOfValid)
  ratio[np.where(ratio==0)] = 0
  ratio[np.isnan(ratio)] = 0
  ratio[np.isinf(ratio)] = 0
 
  return ratio
#
# writeExtremesToFile('highVUIR.txt', ratio, ratio, 'times of high vuir', '  ');
# writeExtremesToFile('lowVUIR.txt', -ratio, ratio, 'times of low vuir',  '  ');
#
# clf
# hold on 
# plot(1:length(pitch), 100 * isSpeech, ...
#      1:length(pitch), 10 * logEnergy, ...
#      1:length(pitch), pitch, ...
#      1:length(pitch), 100 * ratio);
#   legend('isSpeech', 'logEnergy', 'pitch', 'v-uv i ratio');



#test data
#  silence energy around 1, voiced around 9,
#quiet unvoiced around 5, loud unvoiced around 7
#
#testdata set 1 represents silence then a vowel
#testdata set 2 represents an quiet unvoiced consonant then a vowel 
#testdata set 3 represents silence then a loud unvoiced consonant
#
#p1 =np.array( [5,np.nan,5,np.nan,6,np.nan,np.nan,np.nan,np.nan,7,8,7,7,1,3,2,8,3,9,np.nan])
#e1 = np.array([1,1,0,1,0,1,0,2,0,9,9,8,8,9,7,2,9,8,8,8])
##p2 = [2 NaN NaN NaN NaN NaN NaN NaN 7 8 7 8 7 8 9 8 9 7];
##e2 = [4   5   4   5   3   5   7   4 8 9 1 8 9 9 8 7 6 9];
##p3 = [2 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN ];
##e3 = [4   0   1   1   2   1   0   1   2   7   8   7   7   7   7 ];
##
#ratio=voicedUnvoicedIR(e1, p1, 200)  # ratio result should be around 1.8
##voicedUnvoicedIr([e1 e3], [p1 p3], 200);  % ratio result should be around 1.3
#np.set_printoptions(threshold=np.nan)
#print(ratio)
#I also "tested" it by running it on 21d and seeing where the value is high/low
#It seemed high during politer regions, and low during self-talk.
