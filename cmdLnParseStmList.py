# -*- coding: utf-8 -*-
"""
Created on Wed Aug 07 14:17:56 2019


@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA        
"""
"""
Function to run the parseStm.py from the command prompt
which gets tha path to the file from an input list.
"""
import sys
import parseStm
import os

def main():
    script = sys.argv[0]
    stmfileDir= sys.argv[1]
    stmfileList= sys.argv[2]
    
    filesOnlyInstmfileDir = (file for file in os.listdir(stmfileDir) 
    #check if it is a file #and not a sub-directory
         if os.path.isfile(os.path.join(stmfileDir, file)))
    
    with open(stmfileList) as sl:
      stmList=sl.read().splitlines()  
                                            
    for stmfile in filesOnlyInstmfileDir:
        
        for stm in stmList:
            if(stmfile==stm):                
#          print('It is an stmfile')
#          print(stmfile)
              stm_segments,doctor_segments,patient_segments=\
              parseStm.parse_stm_file(stmfileDir + '/' + stmfile)
              
if __name__== "__main__":
  main()