import numpy as np
from findClusterMeans import findClusterMeans
def windowEnergy(logEnergy,msPerWindow):
    #inputs:
    #   logenergy: logenergy over every 10 millisecond frame,
    #   first one centered at 5ms, next at 15 ms, and so on
    #   msPerWindow: duration of window to compute energy over
    #output:
    #   energy over windows of the designated size, cented at 10ms, 20ms, etc.
    #   (the centering is off, at 15ms, 25ms, etc, if msPerWindow is 30ms, 50ms etc)
    #   but we're not doing syllable-level prosody, so it doesn't matter.
    #   Values returned are zero if either end of the window would go outside
    #    what we have data for.

    #   Nigel Ward, UTEP, December 2014, Feb 2015, July 2017


    integralImage = np.concatenate((np.zeros(1),np.cumsum(logEnergy)),axis=0)

    framesPerWindow = msPerWindow / 10

    windowSum = integralImage[int(framesPerWindow):] - integralImage[:integralImage.size - int(framesPerWindow)]
    # find silence and speech mean of track using k-means, then use them
    # to normalize for robustness against recording volume
    [silence_mean, speech_mean] = findClusterMeans(windowSum)
    difference = speech_mean - silence_mean
    #print(difference)
    if difference > 0:
         scaledSum = (windowSum - silence_mean) / difference
    else:
        # then something's wrong; typically the file is mostly music or
        # has a terribly low SNR.  So we just return something that at
        # least has no NaNs tho it may or may not be useful.
         scaledSum = (windowSum - (0.5 * silence_mean)) / silence_mean
    #align so first value is for window centered at 10 ms (or 15ms if, an odd number of frames)
    #using zeros for padding is really not right
    headFramesToPad = np.int(np.floor(framesPerWindow / 2) - 1)
    tailFramesToPad = np.int(np.ceil(framesPerWindow / 2) - 1)
    #horzcat
    winEnergy = np.append(np.zeros(headFramesToPad),scaledSum)
    winEnergy = np.append(winEnergy,np.zeros(tailFramesToPad))
    return winEnergy

# test cases:
#print(windowEnergy([0,1,1,1,2,3,3,3,1,1,1,2], 20))
# windowEnergy([0,1,1,1,2,3,3,3,1,1,1,2], 30)
#
# [r,s] = readtrack('../minitest/short.au', 'l')
# e = computeLogEnergy(s(:,1)', 80, 80)
# w = windowEnergy(e, 20)
# plot(20 * (1:length(w)), w, 'g')
# hold
# w = windowEnergy(e, 50)
# plot(20 * (1:length(w)), w, 'k')
# save logEnergy;