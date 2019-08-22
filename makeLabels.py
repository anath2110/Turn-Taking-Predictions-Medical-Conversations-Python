# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:15:53 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA
"""
"""
Modified on Aug 12, 2019
"""
"""
Function returns vectors of 0s and 1s and 2s 
for gaps between segments, speech within segments and turn ends, respectively.
"""
from readwav import readwav
import numpy as np
import scipy.io as sio
from parseStm import parse_stm_file

def makeLabels(audioPath,stmFilePath):
    
    [rate,audiodata,channels,audio_duration]= readwav(audioPath)
    #print(audio_duration)
    window_sec = 0.01
    
    
    targets = np.zeros(int((audio_duration/window_sec)))
    #print(targets.shape)
    #print(targets)
    
    
    stm_segments,doctor_segments,patient_segments=parse_stm_file(stmFilePath)
   
    for seg in range(len(doctor_segments)):
        startSeg=doctor_segments[seg].start_time
        #print(startSeg)
        endSeg=doctor_segments[seg].stop_time
        #print(endSeg)
        
        #each of entire doctor segment labeleld as '1'
        for windowedSeg in np.arange(startSeg,endSeg,window_sec):
                labelindex=int(windowedSeg/window_sec)
                #print(labelindex)
                if(labelindex <targets.shape[0]):
                    targets[labelindex]=1      
      
        turnindex=int(endSeg/window_sec)      
        #print(labelindex)
        if(turnindex <targets.shape[0]):
            #print(turnindex)
            targets[turnindex]=2 # turn ends are 2
         #only the end of doctor turn is labelled as '2'  
    
    
    """ Same as above is repeated for patient turn"""
    for seg in range(len(patient_segments)):
        startSeg=patient_segments[seg].start_time
        endSeg=patient_segments[seg].stop_time
        #print(endSeg)
        #each of entire patient segments labeleld as '1'
        for windowedSeg in np.arange(startSeg,endSeg,window_sec):
                labelindex=int(windowedSeg/window_sec)
                if(labelindex <targets.shape[0]):
                    targets[labelindex]=1
  
        #print(labelindex)
        if(turnindex <targets.shape[0]):
            #print(turnindex)
            targets[turnindex]=2 # turn ends are 2
         #only the end of doctor turn is labelled as '2' 
#    saveas='targets'+ stmFilePath[len(stmFilePath)-10:] + '.mat'#10==len(100002.stm)
#    #print(saveas)
#    sio.savemat(saveas, {'targets': targets})
#        #Note: in .mat file, indexes are added by one since python indexing 
#        #starts from '0'
#    #print(targets.shape)
    return targets

#print(makeLabels('D:/FUNDataCodes/Audios/sampleAudios/100039.wav','D:/FUNDataCodes/NewTurnStm/sampleStms/100039.stm'))