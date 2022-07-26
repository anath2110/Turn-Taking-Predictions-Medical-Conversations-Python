# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:14:39 2019
@author: Anindita Nath
3M| M*Modal
Pittsburgh, PA
"""
"""
Only change from lookOrComputePitchWav:Imports amfm_decompy.pYAAPT_MATLABverForWav as pYAAPT
"""
#from __future__ import unicode_literals
from datetime import datetime
#import pickle
#import _pickle as pickle #cPickle version in Python3.x
#import hdf5storage
import os.path
import numpy as np
import scipy
from pathlib import Path
#import amfm_decompy.pYAAPT as pYAAPT #original code
import amfm_decompy.pYAAPT_MATLABverForWav as pYAAPT 



def  lookOrComputePitchWav(pathToSavePitchCache,directory,audio,side,signal):    

#Savekey encodes the audio filename and the track.
#If a cached pitch file exists, then use that data 
#otherwise compute pitch and save it 
#Return a vector of pitch points and a vector of where they are, in ms

    
    savekey=audio + side
    pitchCacheDir = pathToSavePitchCache + 'pitchCachePython'
    
    if not os.path.exists(pitchCacheDir):
        os.makedirs(pitchCacheDir)  
   
    #name of the pitch file in .mat format
    pitchFileName = pitchCacheDir + '/pitchpython' + savekey + '.mat'
    #pitchfiledata={} # dictionary to save pitch in hdf5 format
    #name of the pitch file in pickle format
    #pitchFileName = pitchCacheDir + '/pitchpython' + savekey + '.pkl'
    
    if (Path(pitchFileName).is_file()==False):
        
    # pitch mat file doe not exist   
      print('computing pitch for ' + savekey)
     
      if (signal.channels==2): # if a stereo
          if(side=='l'):# take signal values of left track 
              signal.data=signal.data[:,0]
          elif(side=='r'):# take signal values of right track 
              signal.data=signal.data[:,1]
      
      pitch = pYAAPT.yaapt(signal)
      pitchvals=pitch.samp_values
      #print(pitchvals.shape)
      pitch_centres = pitch.frames_pos
      time_stamp_in_seconds = pitch_centres/signal.fs #divide by msperframe, matlab compatibility
         
      
      #save pitch as .mat        
#      scipy.io.savemat(pitchFileName, {'pitchpy': pitch,'pitchsamples': pitchvals,\
#      'pitch_centres': pitch_centres,'time_stamp_in_seconds': time_stamp_in_seconds})
      scipy.io.savemat(pitchFileName, {'pitchsamples': pitchvals,\
      'pitch_centres': pitch_centres,'time_stamp_in_seconds': time_stamp_in_seconds})
      
      # following can store in version >=7.3, needed for large .mat files
      #'appendmat' appends'.mat' extension,truncate_existing=False appends to existing .mat file else overwrites 
      #pitchfiledata[u'pitchpy'] = pitch # cannot save class objects in hdf5storage
#      pitchfiledata[u'pitchsamples'] = pitchvals
#      pitchfiledata[u'pitch_centres'] = pitch_centres
#      pitchfiledata[u'time_stamp_in_seconds'] = time_stamp_in_seconds
#      hdf5storage.savemat(pitchFileName,pitchfiledata ,appendmat=True,format='7.3',truncate_existing=False)
#          
          
      
    
     #save pitch as pickle file
     #with open(pitchFileName, 'wb') as outfile:
          #pickle version
          #pickle.dump([pitchvals,pitch_centres,time_stamp_in_seconds],outfile, pickle.HIGHEST_PROTOCOL)
          #cPickle version
          #pickle.dump([pitchvals,pitch_centres,time_stamp_in_seconds],outfile, protocol=3)
    
    else:
      if file1isOlder(pitchFileName, directory + audio):    
        print('recomputing pitch for' +  savekey)
        
        if (signal.channels==2): # if a stereo
          if(side=='l'):# take signal values of left track 
              signal.data=signal.data[:,0]
          elif(side=='r'):# take signal values of right track 
              signal.data=signal.data[:,1]
        
        pitch = pYAAPT.yaapt(signal)
        pitchvals=pitch.samp_values
        pitch_centres = pitch.frames_pos
        time_stamp_in_seconds = pitch_centres/signal.fs #divide by msperframe, matlab compatibility
        
        # save pitch as .mat  
#        scipy.io.savemat(pitchFileName, {'pitchpy': pitch,'pitchsamples': pitchvals,\
#        'pitch_centres': pitch_centres,'time_stamp_in_seconds': time_stamp_in_seconds}) 
        scipy.io.savemat(pitchFileName, {'pitchsamples': pitchvals,\
        'pitch_centres': pitch_centres,'time_stamp_in_seconds': time_stamp_in_seconds}) 
    #following can store in version >=7.3, needed for large .mat files
      #'appendmat' appends'.mat' extension,truncate_existing=False appends to existing .mat file else overwrites 
        #pitchfiledata[u'pitchpy'] = pitch # hdf5 cannot store class objects
#        pitchfiledata[u'pitchsamples'] = pitchvals
#        pitchfiledata[u'pitch_centres'] = pitch_centres
#        pitchfiledata[u'time_stamp_in_seconds'] = time_stamp_in_seconds
#        hdf5storage.savemat(pitchFileName,pitchfiledata ,appendmat=True,format='7.3',truncate_existing=False)
#          
        #save pitch as pickle file
        #with open(pitchFileName, 'wb') as outfile:
          #pickle version
          #pickle.dump([pitchvals,pitch_centres],outfile, pickle.HIGHEST_PROTOCOL)
          #cPickle version
          #pickle.dump([pitchvals,pitch_centres,time_stamp_in_seconds],outfile, protocol=3)
    

      else: 
        print('reading cached pitch file '+ pitchFileName)
        #read .mat pitch files from cache
        pitchpy=scipy.io.loadmat(pitchFileName)

        #pitchpy=hdf5storage.loadmat(pitchFileName)

        pitchsamples=pitchpy['pitchsamples'] 
        pitch_centres = pitchpy['pitch_centres']  
        pitchvals = np.array(pitchsamples)
        pitch_centres = np.array(pitch_centres)
#        print(pitchsamples.shape)   
        
        #read .pkl pitch files from cache
#        with open(pitchFileName, 'rb') as f:
#            pitch = pickle.load(f)
#            
#        pitchvals = np.array(pitch[0]) #each variable stored is numbered sequentially in pitch       
#        pitch_centres = np.array(pitch[1])
        
 

    
    #The first pitch point yaapt returns is for a frame from 17.5ms to 27.5ms, 
    #thus centered at 20ms into the signal.
    #The last one is similarly short of the end of the audio file. 
    #So we pad. 
    
    paddedPitch = np.insert(pitchvals,0,np.nan)
    paddedPitch=np.append(paddedPitch,np.nan)
    
   
    #we know that pitchpoints are 80 milliseconds apart
    paddedCenters = np.insert(pitch_centres,0,pitch_centres[0] - 80) 
    paddedCenters=np.append(paddedCenters,pitch_centres[len(pitch_centres)-1] + 80)
    
    
    return paddedPitch,paddedCenters
   


#------------------------------------------------------------------
def file1isOlder(file1, file2):
    try:
        file1time = os.path.getmtime(file1)
        #print(file1time)
        file2time = os.path.getmtime(file2)
        #print(file2time)
    except OSError:
        file1time = 0
        file2time = 0
        file1time = datetime.fromtimestamp(file1time)
        file2time = datetime.fromtimestamp(file2time)
        #print(file1time)
        #print(file2time)
  
    isOlder = file1time < file2time
    #print(isOlder)
    return isOlder


#test 
#signal = austruct.SignalObj('../testeng/audio/2ndWeekendNewscastJuly292012.au')
#paddedPitch,paddedCenters=lookupOrComputePitch('../testeng/audio/','2ndWeekendNewscastJuly292012.au','l',signal)
#print(paddedPitch.shape)
#paddedPitch,paddedCenters=lookupOrComputePitch('stance-master/testeng/audio/','f0a_01.au','l')