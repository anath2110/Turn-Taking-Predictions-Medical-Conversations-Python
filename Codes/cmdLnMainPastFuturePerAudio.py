# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:17:56 2019


@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA        
"""
"""
Function to run the featuresPastPresentFuturePerAudio.py from the command prompt.
"""
import sys

import featuresPastPresentFuturePerAudio
from getDialogsList import getDialogsList


def main():
    script = sys.argv[0]
    dialogListFile=sys.argv[1]    
    pathToSavePitchCache=sys.argv[2]
    pathToSaveFeatsLabels=sys.argv[3]
    audioDirPath=sys.argv[4]
    labelsDirPath=sys.argv[5]
    split= float(sys.argv[6])
    
    audioFiles=getDialogsList(dialogListFile)
  
    for audioIndex in range(0,len(audioFiles)):
        audio=audioFiles[audioIndex] #audio base name with extension 
        
        features, labels=\
        featuresPastPresentFuturePerAudio.featuresAndLabels(\
                                                   pathToSavePitchCache,
                                             pathToSaveFeatsLabels,audio,
                                            audioDirPath,labelsDirPath,split)
if __name__== "__main__":
  main()