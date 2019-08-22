from __future__ import division
import numpy as np
from readtracks import readtracks
from computeLogenergy import computeLogEnergy
import matplotlib.pyplot as plt
from base import mfcc
from scipy.signal import medfilt
#Translated on Python by Alonso Granados 11/10/2017
def cepstralFlux(signal, rate, energy):
    # Nigel Ward, UTEP, December 2016
    # Testing on.. / flowtest / prefix21d.au, I observe that:
    #  - this is strongly high when the speaking rate is high
    #  - and also during very creaky regions, even if they sound lengthened,
    #  - very low during silence
    #  - generally moderately low during lengthenings
    #  computeLengthening() the flux returned by this function
    signal = np.array(signal)
    # mfcc parameters taken straight from Kamil Wojcicki's mfcc.m documentation
    # James Lyons mfcc implementation
    cc = mfcc(signal, rate, 0.025, 0.01, 13,20,512,300,3700,0.97,22,False,winfunc=np.hamming)
    #print(cc.shape)
    cc = np.concatenate((np.zeros((1,13)),cc))
    cc = np.concatenate((cc,np.zeros((1,13))))
    #print(cc.shape)
    smoothingSize = 15  # smooth over 150ms windows
    diff = cc[1:,:] - cc[:cc.shape[0] -  1,:]
    #print(diff.shape)
    if len(diff) < len(energy):
        # pad it with a convenient value
        avgdiff = np.mean(abs(diff),axis=0)
        diff = np.vstack((avgdiff, diff))
    elif len(diff) > len(energy):
        diff=diff[:len(diff)-1]
    else:
        diff=diff
    diffSquared = diff*diff
    sumdiffsq = np.sum(diffSquared,axis=1)
    # the function smooth is only in the latest Matlab release,
    # but there's an alternative implmentation in smoothJcc.m
    sumdiffsq = medfilt(sumdiffsq, kernel_size=smoothingSize)
    #plt.plot(sumdiffsq)
    #plt.show()
    return sumdiffsq

#energy=np.array(energy=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
#[s,c,r] = readtracks('stance-master/testeng/audio/ChevLocalNewsJuly3.au')
#samplesPerFrame = int(10 * int(r / 1000))
#e = computeLogEnergy(s, samplesPerFrame)

#[signals,channels,rate]=readtracks.readtracks('21d.au')
#signals = signals[::2]
#print("Signals")
#print(signals[0:10])
#print("Rate")
#print(rate)
#newEnergy =computeLogenergy.computeLogEnergy(signals, 80)
#print("Energy")
#print(newEnergy[0:10])
#print("Cepstral")
#rate = 16000
#flux=cepstralFlux(s,r,e)
#print(e.shape)
##print(s.shape)
#print(flux.shape)

