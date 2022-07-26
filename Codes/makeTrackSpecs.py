# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:35:06 2017

@author: Anindita Nath
"""
#from scipy import io
from collections import namedtuple

def makeTrackSpecs(sideval, filenameval, directoryval):
  
  trackspec=namedtuple("trackspec","side filename directory path")
  pathval =directoryval + filenameval
  ts= trackspec(sideval, filenameval, directoryval,pathval)
  #print(ts.side)
  return ts
  
###Test###
#ts=makeTrackSpecs("l", "aaaa.au", "sddffsdf/")
#print(ts)
#savemat('c:/tmp/arrdata.mat', mdict={'arr': arr})
