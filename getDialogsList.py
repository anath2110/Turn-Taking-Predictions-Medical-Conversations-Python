# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:08:29 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA  

"""
""""  
Function takes the list of dialogs, .txt file,  as input 
and returns it  as a numpy array.
"""      

import numpy as np
import os


def getDialogsList(dialogListFile): 
  
  dialogNames=[]
  
  
  with open(dialogListFile) as dl:
      dialogList=dl.read().splitlines()
      
      #print(dialogList)      
  
  for dialogIndex in range(len(dialogList)):
       #print(dialogList[dialogIndex])
       if (dialogList[dialogIndex].endswith('.wav')):
           dialogNames.append(dialogList[dialogIndex])
        
  dialogNames=np.asarray(dialogNames)
    
  return dialogNames

#print(getDialogsList('D:/FUNDataCodes/OAKLists/OAK1_2testDialogsAudios.txt'))
