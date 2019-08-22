# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:50:37 2017

@author: Anindita Nath
University of Texas at ElPaso
"""
import numpy as np
import math
import scipy
import sys
import warnings
import readau_structobject as austruct
import matplotlib.pyplot as plt
from getfeaturespec import getfeaturespec
from makeTrackSpecs import makeTrackspec
from featurizegaze import featurizeGaze
from featurizeKeystrokes import featurizeKeystrokes
#from readtracks import readtracks
from computeLogenergy import computeLogEnergy
from lookOrComputePitch import lookupOrComputePitch
from cepstralFlux import cepstralFlux
from killBleeding import killBleeding
from percentilizePitch import percentilizePitch
from windowize import windowize
from findClusterMeans import findClusterMeans
from distanceToNearness import distanceToNearness
from computeCreakiness import computeCreakiness
from windowEnergy import windowEnergy
from voicedUnvoicedIR import voicedUnvoicedIR
from cepstralDistinctness import cepstralDistinctness
from computeRate import computeRate
from speakingFraction import speakingFraction
from computePitchInBand import computePitchInBand
from computePitchRange import computePitchRange
from computeWindowedSlips import computeWindowedSlips
from computeLengthening import computeLengthening

def makeTrackMonster(trackspec, featurelist):

#inputs:
#   trackspec: includes au pathname plus track (left or right)
#   fsspec: feature set specification
# output:
#   monster is a large 2-dimensional array, 
#     where every row is a timepoint, 10ms apart, starting at 10ms (?)
#    and every column a feature
#    firstCompleteFrame, is the first frame for which all data tracks
#    are present.   This considers only the fact that the gaze
#     track may not have values up until frame x
#     It does not consider the fact that some of the wide past-value
#     features may not have meaningful values until some time into the data
#     The reason for this is that, in case the gaze data starts late,
#     we pad it with zeros, rather than truncating the audio.  This is because
#     we compute times using not timestamps, but implicitly, e.g. frame 0 
#     is anchored at time zero (in the audio track)
#     efficiency issues: 
#   lots of redundant computation
#   compute everything every 10ms, then in the last step downsample to 20ms
# testing:
#   the simplest test harness is validateFeature.m

#converted from original code by Nigel Ward, UTEP, 2014-2015

    plotThings = False
    processGaze = False
    processKeystrokes = False  
    processAudio = False
    firstCompleteFrame = 1
    lastCompleteFrame = 9999999999999
    
    for featureNum in range(len(featurelist)):
       thisfeature = featurelist[featureNum]
       
       if np.in1d(thisfeature.featname, np.array(['ga', 'gu', 'gd', 'gl', 'gr', 'go'])):
           processGaze = True
      
       if  np.in1d(thisfeature.featname, np.array(['rf', 'mi', 'ju'])):
        	processKeystrokes = True
       
       if  np.in1d(thisfeature.featname,np.array( ['vo', 'th', 'tl', 'lp', 'hp', 'fp', \
       'wp', 'np', 'sr', 'cr', 'pd', 'le', 'vf', 'sf', 're', 'en', 'ts', 'te'])):
        	processAudio = True
    
    
    if (processGaze==True):
       [ssl, esl, gzl, gul, gdl, gll, grl, gfl] = \
           featurizeGaze(trackspec.path, 'l')
       [ssr, esr, gzr, gur, gdr, glr, grr, gfr] = \
           featurizeGaze(trackspec.path, 'r')
       firstFullyValidTime = max(ssl, ssr)  # presumably always > 0
       firstCompleteFrame = int(math.ceil(firstFullyValidTime * 100))
       lastCompleteFrame = min(len(gzl), len(gzr))
    
    
    if (processKeystrokes ==True) :
       [wrf,wju,wmi] = featurizeKeystrokes(trackspec.path, 'W', 100000)
       [frf,fju,fmi] = featurizeKeystrokes(trackspec.path, 'F', 100000)

    
    msPerFrame = 10
    
    if (processAudio ==True):
      #------ First, compute frame-level features: left track then right track ------
      stereop = decideIfStereo(trackspec, featurelist)
      #print('in monster')
      #print(trackspec.path)
      #[signalPair,channels,rate] = readtracks(trackspec.path)
      signal = austruct.SignalObj(trackspec.path)
      rate=signal.fs
      if (signal.channels < 2 and stereop):
        print('%s is not a stereo file, though the feature list ', trackspec.path);
        print('and/or \n the channel in the trackspec suggested it was.  Exiting\n');
        print('not stereo')
        sys.exit(0)
   
      
      samplesPerFrame = msPerFrame * int(rate / 1000)
      
      if (signal.channels==1):
          signall = signal.data
          plraw, pCenters = lookupOrComputePitch(trackspec.directory,trackspec.filename, 'l',signall)
          energyl = computeLogEnergy(signall, samplesPerFrame)
      #  print('pitch found at %d points\n', np.sum(plraw > 0)) #not-isNan count
      #  print('pitch undefined  at %d points\n', np.sum(isnan(plraw)))
    
      pitchl = plraw 
      cepstralFluxl = cepstralFlux(signall, rate, energyl)
    
      if (stereop==True):
        signalr = signal.data[:,1]
        [prraw, pCenters] = lookupOrComputePitch(\
             trackspec.directory, trackspec.filename, 'r',signal)
        energyr = computeLogEnergy(signalr, samplesPerFrame)
        cepstralFluxr = cepstralFlux(signalr, rate, energyr)
        pitchl, pitchr= killBleeding(plraw, prraw, energyl, energyr)

    nframes = int(math.floor(signal.data.shape[0]/ samplesPerFrame))
    lastCompleteFrame = min(nframes, lastCompleteFrame)
    
    # --- plot left-track signal, for visual inspection ---
    if  (plotThings==True):
      plotEndSec = 8  #plot the first few seconds of the signal and featueres
      yScalingSignal = .005
      yScalingEnergy = 6
      xaxis=np.arrange(int(1/rate),plotEndSec,int(1/rate))
      yaxis=np.arrange(signal.data[:int(rate*plotEndSec),1]* yScalingSignal)
      plt.plot(xaxis,yaxis)
      #plot pitch, useful for checking for uncorrected bleeding
      pCentersSeconds =int( pCenters / 1000)
      pCentersToPlot = pCentersSeconds[pCentersSeconds<plotEndSec]
      plt.scatter(pCentersToPlot, pitchl[:len(pCentersToPlot)], 'g', '.')
      plt.scatter(pCentersToPlot, 0.5 * pitchl[:len(pCentersToPlot)], 'y', '.') # halved
      plt.scatter(pCentersToPlot, 2.0 * pitchl[:len(pCentersToPlot)], 'y', '.') # doubled
      offset = 0  
      plt.scatter(pCentersToPlot, pitchr[:len(pCentersToPlot)] + offset, 'k.')   
      #plot((1:length(energyl)) * msPerFrame, energyl * yScalingEnergy,'g') 
      plt.xlabel('seconds')
 
    maxPitch = 500
    pitchLper = percentilizePitch(pitchl, maxPitch)
    if (stereop==True):
      pitchRper = percentilizePitch(pitchr, maxPitch)
   # print("energyshape")
    #print(energyl.shape[0])
    
    # ------ Second, compute derived features, and add to monster ------
    features_array=np.zeros(shape=(energyl.shape[0]-1,len(featurelist)))
    if (stereop==True):
        features_array=np.zeros(shape=(energyr.shape[0]-1,len(featurelist)))
    #features_array=[]
    for featureNum in range(len(featurelist)):
      thisfeature = featurelist[featureNum]
      duration = thisfeature.duration
      startms = thisfeature.startms
      endms = thisfeature.endms
      feattype = thisfeature.featname
      side = thisfeature.side
      plotcolor = thisfeature.plotcolor
    
      if (processAudio==True):
        if ((side =='self') and (trackspec.side =='l')) or \
           ((side =='inte') and (trackspec.side =='r')):
              relevantPitch = pitchl
              relevantPitchPer = pitchLper
              relevantEnergy = energyl
              relevantFlux = cepstralFluxl
              relevantSig = signall
              [lsilenceMean, lspeechMean] = findClusterMeans(energyl)
        else: 
          #if stereop is False then this should not be reached 
          relevantPitch = pitchr
          relevantPitchPer = pitchRper
          relevantEnergy = energyr
          relevantFlux = cepstralFluxr
          relevantSig = signalr
          [rsilenceMean, rspeechMean] = findClusterMeans(energyr)
     
      if (processGaze==True):
        if ((side=='self') and (trackspec.side=='l')) or \
    	  ((side =='inte') and (trackspec.side =='r')):
          relGz = gzl
          relGu = gul
          relGd = gdl
          relGl = gll
          relGr = grl
          relGa = gfl
        else:
          relGz = gzr
          relGu = gur
          relGd = gdr
          relGl = glr
          relGr = grr
          relGa = gfr
        
      if (processKeystrokes==True):
        if ((side=='self') and (trackspec.sprite =='W')) or\
           ((side =='inte') and (trackspec.sprite =='F')):
           relevantJU = wju
           relevantMI = wmi
           relevantRF = wrf
        else:
           relevantJU = fju
           relevantMI = fmi
           relevantRF = frf
    
    
    #print('processing feature %s %d %d %s \n', ...
    #	  feattype, thisfeature.startms, thisfeature.endms, side)
        
      if (feattype=='vo'): #volume/energy/intensity/amplitude
          featurevec = windowEnergy(relevantEnergy, duration)  
         # print('vo' + str(featurevec.shape))
          #print(featurevec)
      elif (feattype=='vf'):   # voicing fraction
          relevantPitch=np.insert(relevantPitch,0,0)
          relevantPitch=np.append(relevantPitch,0)
          featurevec = windowize(~np.isnan(relevantPitch), duration)
         # print('vf' + str(featurevec.shape))
      elif (feattype=='sf'):     # speaking fraction
          featurevec = speakingFraction(relevantEnergy, duration)
          #print('sf' + str(featurevec.shape))
      elif (feattype=='en'):     #cepstral distinctiveness
          featurevec = cepstralDistinctness(relevantSig, rate, relevantPitch, duration, 'enunciation')
          #print('en' + str(featurevec.shape))
      elif (feattype=='re'):    #cepstral blandness
          featurevec = cepstralDistinctness(relevantSig, rate, relevantPitch, duration, 'reduction')
          #print('re' + str(featurevec.shape))
      elif (feattype=='th'):       #pitch truly high-ness
          featurevec = computePitchInBand(relevantPitchPer, 'th', duration)
          #print('th' + str(featurevec.shape))
      elif (feattype=='tl'):  #pitch truly low-ness
          featurevec = computePitchInBand(relevantPitchPer, 'tl', duration)
         # print('tl' + str(featurevec.shape))
      elif (feattype=='lp'):     # pitch lowness 
          featurevec = computePitchInBand(relevantPitchPer, 'l', duration)
          #print('lp' + str(featurevec.shape))
      elif (feattype=='hp'):   # pitch highness
          featurevec = computePitchInBand(relevantPitchPer, 'h', duration)
         # print('hp' + str(featurevec.shape))
      elif (feattype=='fp'):    # flat pitch range 
          featurevec  = computePitchRange(relevantPitch, duration,'f')
          #print('fp' + str(featurevec.shape))
      elif (feattype=='np'):    #narrow pitch range 
          featurevec  = computePitchRange(relevantPitch, duration,'n')
         # print('np' + str(featurevec.shape))
      elif (feattype=='wp'):    #wide pitch range 
          featurevec  = computePitchRange(relevantPitch, duration,'w') 
        #  print('wp' + str(featurevec.shape))
      elif (feattype=='sr'):    #speaking rate 
          featurevec = computeRate(relevantEnergy, duration)
         # print('sr' + str(featurevec.shape))
      elif (feattype=='cr'):      #creakiness
          featurevec = computeCreakiness(relevantPitch, duration)
        #  print('cr' + str(featurevec.shape))
      elif (feattype=='pd'):    #peakDisalignment
          featurevec = computeWindowedSlips(relevantEnergy, relevantPitchPer, duration)
        #  print('pd' + str(featurevec.shape))
      elif (feattype=='le'):     #lengthening
          featurevec = computeLengthening(relevantEnergy, relevantFlux, duration)
        #  print('le' + str(featurevec.shape))
      elif (feattype=='vr'):     #voiced-unvoiced energy ratio
          featurevec = voicedUnvoicedIR(relevantEnergy, relevantPitch, duration)
        #  print('vr' + str(featurevec.shape))
    
      elif (feattype=='ts'):   #time from start
          featurevec =  windowize(relevantPitch, duration)
         # print('ts' + str(featurevec.shape))
      elif (feattype=='te'):    #time until end
          featurevec =  windowize(np.subtract(len(relevantPitch),relevantPitch), duration)
         # print('te' + str(featurevec.shape))
    
      elif (feattype=='ns'):   #near to start
          featurevec = distanceToNearness(windowize(relevantPitch, duration))
         # print('ns' + str(featurevec.shape))
    
      elif (feattype=='ne'): #near to end
          featurevec = distanceToNearness(windowize(np.subtract(len(relevantPitch),relevantPitch), duration))
         # print('ne' + str(featurevec.shape))
    
      elif (feattype=='rf' ):   #running fraction
          featurevec = windowize(relevantRF, duration)  # note, transpose
        #  print('rf' + str(featurevec.shape))
      elif (feattype=='mi'):      #motion initiation
          featurevec = windowize(relevantMI, duration) #note, transpose
         # print('mi' + str(featurevec.shape))
      elif (feattype== 'ju'):      #jumps
          featurevec = windowize(relevantJU, duration)  #note, transpose
        #  print('ju' + str(featurevec.shape))
    
      elif (feattype== 'go'): 
          if duration == 0:         #then we just want to predict it 
              featurevec = relGz[:len(featurevec)-1]
              
          else:
              featurevec = windowize(relGz, duration) 
          #print('go' + str(featurevec.shape))
      elif (feattype== 'gu'):
          featurevec = windowize(relGu, duration)
          #print('gu' + str(featurevec.shape))
      elif (feattype== 'gd'): 
          featurevec = windowize(relGd, duration)
          #print('gd' + str(featurevec.shape))
      elif (feattype== 'gl'):  
          featurevec = windowize(relGl, duration)
          #print('gl' + str(featurevec.shape))
      elif (feattype== 'gr'):   
          featurevec = windowize(relGr, duration)
          #print('gr' + str(featurevec.shape))
      elif (feattype== 'ga'):   
          featurevec = windowize(relGa, duration) 
          #print('ga' + str(featurevec.shape))
    
      else:
          print("Warning!" + feattype + ' :  unknown feature type')

      #print(featurevec.shape)
      h= len(featurevec.shape)
     
      
      
    #time-shift values as appropriate, either up or down (backward or forward in time)
    #the first value of each featurevec represents the situation at 10ms or 15ms
      centerOffsetMs = int((startms + endms) / 2)     #offset of the window center
      shift = int(round(centerOffsetMs / msPerFrame))
      if (shift == 0):
        shifted = featurevec
      elif (shift < 0):
        #then we want data from the past, so move it forward in time, 
        #by stuffing zeros in the front 
       
        shifted = np.concatenate((np.zeros(-shift,),featurevec[:h+shift]),axis=0)
      else:
        #shift > 0: want data from the future, so move it backward in time,
        #padding with zeros at the end
        shifted = np.concatenate((featurevec[shift+1:], np.zeros(shift,)))
     
    
      if (plotThings and plotcolor != 0):
         plt.plot(pCentersToPlot, featurevec[:len(pCentersToPlot)] * 100, plotcolor)
      
      #might convert from every 10ms to every 20ms to same time and space,
      #here, instead of doing it at the very end in writePcFilesBis
      #downsampled = shifted(2:2:end);   
     
      shifted = shifted[:lastCompleteFrame-1]
      
      if (len(shifted)==(energyl.shape[0]-1)):
          features_array[:,featureNum]=shifted 
      elif(len(shifted)>energyl.shape[0]-1):
          shifted=shifted(len(shifted)-1)
      else:
          #shifted=np.append(shifted,np.zeros(1,))
          shifted=np.insert(shifted,0,0)
          
          
      #features_array.append(shifted)#append shifted values to monster
      
      #loop back to do the next feature
      
    monster = features_array  #flatten it to be ready for princomp


    return firstCompleteFrame, monster

#this is tested by calling findDimensions for a small audio file (e.g. short.au)
#and a small set of features (e.g. minicrunch.fss)
#and then uncommenting various of the "scatter" commands above
#and examining whether the feature values look appropriate for the audio input


#True if trackspec is a right channel or any feature is inte
def decideIfStereo(trackspec, featurelist):
  stereop = False
  if (trackspec.side=='r'):
    stereop = True
  
  for featureNum in range(len(featurelist)):
    thisfeature = featurelist[featureNum]
    if (thisfeature.side=='inte'):
      stereop = True
      
  return stereop 

#flist=getfeaturespec('../testeng/featurefile/mono4.fss')
#ts=makeTrackspec("l", "ChevLocalNewsJuly3.au", "../testeng/audio/")
#fcframe,monster=makeTrackMonster(ts, flist)
#np.set_printoptions(threshold=np.inf)
#scipy.io.savemat('monsterpy.mat', {'monsterpy': monster})
#print(monster)
#print(monster.shape)
#for i in range( len(monster)):
#    print(monster[i].shape)