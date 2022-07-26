# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:17:56 2019


@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA        
"""
"""
Function to run the delExistingMatData.py from the command prompt.
"""
import sys
import parseStm
import os
import delExistingMatData


def main():
    script = sys.argv[0]
    storedMatfileDir= sys.argv[1] 
    
   
    delExistingMatData.delExisting(storedMatfileDir)

main()
