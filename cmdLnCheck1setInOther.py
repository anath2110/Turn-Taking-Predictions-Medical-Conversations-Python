# -*- coding: utf-8 -*-
"""
Created on Wed Aug 07 13:17:56 2019


@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA        
"""
"""
Function to run the checkWhetherInDialogList.py from the command prompt.
"""
import sys
import checkWhetherInDialogList

def main():
    script = sys.argv[0]
    dialoListpath= sys.argv[1]
    audioListpath= sys.argv[2]
 
    
    matchedaudios=\
    checkWhetherInDialogList.checkWhetherInDialogList(dialoListpath,audioListpath)

        
if __name__== "__main__":
  main()