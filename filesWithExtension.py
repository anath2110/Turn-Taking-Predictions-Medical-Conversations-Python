# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:17:55 2017
Modified on Mon Jul 15 11:16:55 2019

@author: ANINDITA Nath
University of Texas at El Paso
"""
import numpy as np
import os

def filesWithExtension(dirpath, extension): 

  filenames = []
  filesOnly = (file for file in os.listdir(dirpath) 
         if os.path.isfile(os.path.join(dirpath, file)))#check if it is a file
                                                     #and not a sub-directory
  for file in filesOnly:
    if file.endswith(extension):        
        filenames.append(file)
  filenames=np.asarray(filenames)
    
  return filenames
#filenames=filesWithExtension("D:/FUNDataCodes/Audios", "wav")
#print(len(filenames))
#print (filenames)