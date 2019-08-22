# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:31:40 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA        
Modified from loadData.py
"""
"""
Function deletes variables from existing .mat files.
"""

import scipy.io as sio
import os 


def delExisting(pathToDirStoredMat):   
    
    filesOnly = (file for file in os.listdir(pathToDirStoredMat) 
         if os.path.isfile(os.path.join(pathToDirStoredMat, file)))#check if it is a file
                                                     #and not a sub-directory
    for file in filesOnly:      
       if file.endswith('.mat'): 
          
           matfile = sio.loadmat(pathToDirStoredMat + file) 
           try:
               if (matfile["pitchpy"]):
                   del matfile["pitchpy"]
                   sio.savemat(pathToDirStoredMat + file, matfile)
                   print("Deleted from" + pathToDirStoredMat + file)
           except Exception:
              print("No pitchpy in the matfile")
              pass
            
    return

##print("Reading data...")
#delExisting('D:/FUNDataCodes/FeatsLabels/sampleOAK3Dialogs/pitchCachePython/')
