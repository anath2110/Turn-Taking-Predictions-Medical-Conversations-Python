import numpy as np
import scipy
import readtracks
import computeLogenergy
def findClusterMeans(values):
    # Given a set of values that are, hopefully, bimodally distributed,
    # finds the centers of the two clusters by iterative search.
    # Used in particular to compute the silence mean and speech mean for
    # a set of energy values.
    # If the data is not bimodal, equal values for loCenter and highCenter
    # are returned
    maxIterations = 20
    previousLowCenter = min(values)
    previousHighCenter = max(values)
    convergenceThreshold = (previousHighCenter - previousLowCenter)/100

    for i in range(0, maxIterations):
        highCenter = averageOfNearValues(values,previousHighCenter, previousLowCenter)
        lowCenter = averageOfNearValues(values, previousLowCenter, previousHighCenter)
        if((abs(highCenter - previousHighCenter)< convergenceThreshold) & (abs(lowCenter - previousLowCenter) < convergenceThreshold)):
            return lowCenter, highCenter
        previousHighCenter = highCenter
        previousLowCenter = lowCenter
    print("WARNING findClusterMeans exceeded maxIterations without converging")
    #print(previousHighCenter)
    #print(highCenter)
    #print(previousLowCenter)
    #print(lowCenter)
    return lowCenter, highCenter



def averageOfNearValues(values, near_mean, far_mean):
    # returns the averages of all points which are closer to the near mean
    # than to the far mean
    # To save time, approximate by taking a sample of 2000 values.
    # Note 1000 is faster, but A.Nath found that with only 1000 samples,
    # this can fail in rare cases, for example when there is a lot of music,
    # since that can cause the distribution to look unimodal.
    nsamples = 2000
    if len(values) < 2000:
        samples = values
    else:
        #sam = []
        #print(len(values))
        #print(round(len(values)/nsamples))       
        #for i in range(0,len(values),round(len(values)/nsamples)):
            #.append(values[i])
        #samples=np.asarray(sam)
        samples=values[0:len(values):round(len(values)/nsamples)]
        #print(samples)
    nearSample = abs(samples - near_mean)
    farSample = abs(samples - far_mean)
    closerSamples=samples[nearSample<farSample]
    #closerSample = []
    #print(samples.size)
    #for i in range(0, samples.size):
        #if nearSample[i] < farSample[i]:
            #closerSample.append(samples[i])
        
    #print(closerSamples)
    if len(closerSamples) == 0:
        subsetAverage = .9 * near_mean + .1 * far_mean
    else:
        subsetAverage = np.mean(closerSamples)
    return subsetAverage


# test cases
# [a b] = findClursterMeans([1 2 3 2 3 2 3 2 3 2 3 4 6 7 6 7 6 7 6 7 1 9 0 6 6 3])

# [r,s] = readtracks
#e = computeLogEnergy(s(:,1)', 80)
#plot(1:length(e),e)

# [silence,speech] = findClustersMeans(e)
#silence does not end up at the noise floor, but somewhat above it,
#which is probably fine.
#[signals,channels,rate]=readtracks.readtracks('2ndWeekendNewscastJuly292012.au')
#logenergy=computeLogenergy.computeLogEnergy(signals[:], 80)
##print(logenergy)
#scipy.io.savemat('logenergy.mat', {'logenergy': logenergy})   
#[a,b] = findClusterMeans(logenergy)
#print(a,b)