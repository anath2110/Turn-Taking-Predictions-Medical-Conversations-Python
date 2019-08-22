# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:17:56 2019


@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA        
"""
"""
Function to run the parseStm.py from the command prompt.
"""
import sys
import parseStm
import os

def main():
    script = sys.argv[0]
    stmfileDir= sys.argv[1]
    filesOnly = (file for file in os.listdir(stmfileDir) 
         if os.path.isfile(os.path.join(stmfileDir, file)))#check if it is a file
                                                     #and not a sub-directory
    for stmfile in filesOnly:
      if stmfile.endswith('.stm'):       
#          print('It is an stmfile')
#          print(stmfile)
          stm_segments,doctor_segments,patient_segments=\
          parseStm.parse_stm_file(stmfileDir + '/' + stmfile)

main()
