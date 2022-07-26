# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:19:43 2019



@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA  

"""
"""
Function checks whther a input list of audios belong to another input list a dialogs.
""" 

import numpy as np
     
def checkWhetherInDialogList(dialogListFile,audiosListFile):
    
    dialogNames=[]

    
    with open(dialogListFile) as dl:
      dialogList=dl.read().splitlines()
    #print(len(dialogList))
    
    with open(audiosListFile) as al:
      audioList=al.read().splitlines()
    #print(len(audioList)) 
      
    for dialog in range(len(dialogList)):
#        print('All Dialog List')
#        print(dialogList[dialog])
        for audio in range(len(audioList)):
#            print('Audio List')
#            print(audioList[audio])
            if (str(audioList[audio])==str(dialogList[dialog])):
                #print(audioList[audio])
                dialogNames.append(audioList[audio])
            
    #print(len(dialogNames))
    #print(len(nondialogNames))
  
    #print(dialogNames.shape)
    return dialogNames   

#dialogNames=checkWhetherInDialogList('D:/FUNDataCodes/OAKLists/OAK1_2trainStms.txt',
#                               'D:/FUNDataCodes/OAKLists/OAK2train_devStms.txt')  
#dialogNames=checkWhetherInDialogList('D:/FUNDataCodes/OAKLists/all11_16khzStms.txt',
#                               'D:/FUNDataCodes/OAKLists/OAK2train_devStms.txt') 
#dialogNames=checkWhetherInDialogList('D:/FUNDataCodes/OAKLists/OAK1_2train_dev_testNewStms.txt',
#                               'D:/FUNDataCodes/OAKLists/OAK2train_devStms.txt') 
#dialogNames=checkWhetherInDialogList('D:/FUNDataCodes/OAKLists/OAK1_2train_dev_testOldStms.txt',
#                               'D:/FUNDataCodes/OAKLists/OAK2train_devStms.txt') 
