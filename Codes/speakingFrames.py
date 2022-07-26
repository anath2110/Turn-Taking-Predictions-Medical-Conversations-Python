# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:38:18 2017

@author: Anindita Nath
University of Texas at ElPaso
"""
from findClusterMeans import findClusterMeans
import numpy as np
def speakingFrames(logEnergy):
  #returns a vector of 1s and 0s
  # converted from Nigel Ward, UTEP, Feb 2017

  # This is very simplistic, but adequate for the current uses. In
  # future, it might be replaced by  a standard VAD, tho they have a
  # lot of assumptions, or some more sophisticated algorithm.  A
  # simpler change would be to smooth/debounce, since single isolated
  # frames of speech or silence don't exist.

  # find silence and speech mean of track using k-means
  [silenceMean, speechMean] = findClusterMeans(logEnergy)
  
  #Set the speech/silence threshold closer to the silence mean
  #because the variance of silence is less than that of speech.
  #This is ad hoc; modeling with two gaussians would probably be better
  threshold = (2 * silenceMean + speechMean) / 3.0

  vec = logEnergy > threshold
  return vec

#speakingvec=speakingFrames(np.array([0,0,0,0,5,5,5,5,4,4,4,4,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,3,3,3,3,5,5,1,0,0,0]))
#np.set_printoptions(threshold=np.nan)
#print(speakingvec)