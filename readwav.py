# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:25:44 2017
Modified: 06.25.2019 [3M, Pittsburgh,PA]

@author:Anindita Nath 
        University of Texas at ElPaso
"""
import numpy as np
import wave #use this module to read wave files, convinient methods to get 
            #channels amd signal data o/p as user-interpretabel format
import scipy.io as sio
#import scipy.io.wavfile as wf #another class to read wave files 




'''Function to read sound files with .wav extension'''

def readwav(audiofile):
    #print('readwav')
    with wave.open(audiofile, 'r')as f:
    #f=wave.Wave_read(audiofile) #alternative method to open wav file
    #f=wave.open(audiofile)#alternative way, if used have to close the 'f' object at the end
    #rate, audio_dataForWav=wf.read(audiofile) # using scipy class methods to read wav file
        channels=f.getnchannels() #1 for mono, 2 for stereo
        #print(channels)
        rate=f.getframerate()# sampling rate
        #parameters=f.getparams() # outputs all parameters of wav file
        #print(parameters)
        nframes=f.getnframes() # number of frames
        duration=int(nframes/int(rate))
        audio_dataForWav=f.readframes(nframes)        
        #print(audio_dataForWav.shape)
        #np.set_printoptions(threshold=np.nan)
        #print(audio_dataForWav)        
                 
        
        
        #use following if the above is not in the same format as our MATLAB code due to different machine byte encoding
        if(channels==2):
            audio_StringWav = np.ndarray(shape=(nframes,channels),\
                                         dtype=np.int16, buffer=audio_dataForWav) 
#            sio.savemat('audio_StringWavStereo.mat', {'audio_StringWavStereo':\
#                                                       audio_StringWav})
             # output same as MATLAB code, # depends on OS byte format 
             #use either of the following depending which is similar to string format
#            audio_bytearray = np.ndarray(shape=(nframes,channels),dtype='<i2', buffer=audio_dataForWav) 
#           
#            audio_bytearraybigendian = np.ndarray(shape=(nframes,channels),dtype='>i2', buffer=audio_dataForWav) 
#            sio.savemat('audio_byteWavS100002.mat', {'audio_byteWavS100002': audio_bytearray}) #save the byte formats
#            sio.savemat('audio_byteBigWavS100002.mat', {'audio_byteBigWavS100002': audio_bytearraybigendian}) #save the byte formats
#  
        elif(channels==1):
            #following conversion to sting type is depreciated, same as bigByte below 
            #which is preferred as this does not work if audio has 1 channel
            audio_StringWav = np.ndarray(shape=(nframes,),dtype=np.int16, \
                                         buffer=audio_dataForWav) 
#            sio.savemat('audio_StringWavMono.mat', {'audio_StringWavMono':\
#                                                       audio_StringWav})
            # output same as MATLAB code, # depends on OS byte format 
             #use either of the following depending which is similar to string format
#            audio_bytearray = np.ndarray(shape=(nframes,),dtype='<i2', buffer=audio_dataForWav)           
#            audio_bytearraybigendian = np.ndarray(shape=(nframes,),dtype='>i2', buffer=audio_dataForWav) 
#         
#            sio.savemat('audio_byteWavM100002.mat', {'audio_byteWavM100002': audio_bytearray}) #save the byte formats
#            sio.savemat('audio_byteBigWavM100002.mat', {'audio_byteBigWavM100002': audio_bytearraybigendian}) #save the byte formats
   
    #f.close() #needed if audio file opened without 'with'
    return rate,audio_StringWav,channels,duration
    #return rate,audio_dataForWav # when scipy used 


###Test###
    
#[rate,audiodata,channels]=readwav('100002.wav')
#[rate,audiodata,channels,duration]=readwav('D:/FUNDataCodes/Audios/100002.wav')
#[rate,audiodata,channels,duration]=readwav('D:/FUNDataCodes/Audios/ProcessedByPython/100002.wav')
#print(rate,duration)
#[rate,audiodata,channels]=readwav('f10_01.wav')
#[rate,audiodata]=readwav('f10_01.wav')
#np.set_printoptions(threshold=np.nan)
#print(channels,rate)
    
#[rate,audiodata]=readwav('100002.wav')# call scipy class method
#print(rate)



