

<br> ***Aim: To extract prosody features and labels per 10 ms frames over an entire audio(dialog).***

<br>**Run from ssh prompt**:

<br> nohup python nameOfthePythonScript.py pathToATextFileContainingNamesofDialogsOnly.txt pathToSavePitchCachePerAudio PathTOSaveFeaturesAndLabelArraysPerAudio pathToAudoDirectory pathToSTMDirectory split < /dev/null >& customFileNameToStoreStdOutErr.out &

<br> **Input**:

nameOfthePythonScript.py - This the python script to be run from command prompt.
pathToATextFileContainingNamesofDialogsOnly - This is the path to a text file in which each line is the name of a dialog wave file (audiobasename.wav).
pathToSavePitchCachePerAudio - This is the path where the pitch extracted from each of the audio is to be saved. 
PathTOSaveFeaturesAndLabelArraysPerAudio - This is the path where the .mat files containing the feature and label arrays corresponding to each audio is to be saved. 
pathToAudoDirectory - This is the path to the directory containig all the audio files for training/validation/test as the case may be.
pathToSTMDirectory - This is the path to the directory containig all the corresponding stm files(transcriptions).
split- 0 if different train, validation and test set used else the same set is divided into train and test sets in the ratio, train:test = 1-split:split.
customFileNameToStoreStdOutErr.out - Thsi is the filename where all the standard console's output and error get written to.


<br> **Output*:

Returns the feature and label arrays  corresponding to each audio.
Writes the same to disk as 1 .mat file with 2 variables : features and labels containing the corresponding feature and label arrays.


<br> **Sample Runs**:-

Run from /../Codes:

For 6 low-level features(relative pitch i.e. z-normalized pitch, absolute pitch in logHz, energy, cepstral flux, speaking frames from energy and speaking frames from pitch): 
nohup python cmdLnMainPerAudio.py /research2/anindita.nath/OAKLists/OAK2trainDialogsAudios.txt  /local/1/anindita.nath/pitchCacheOAKDialogs8020/ /local/1/anindita.nath/OAK2trainFeatsAndLabels/ /research2/leo.boytsov/data/oak1_2/WAV.all/ /research2/leo.boytsov/pipelines/diarTrain2019-07-03/outStm/ 0 < /dev/null >& OAK2trainSavePerAudioFeatsLabelsNoPitchPy_Aug13.out &

For 64 dimensions LFBE(Log Filter Bank Energy) features: 
nohup python cmdLnMainLFBEPerAudio.py /research2/anindita.nath/OAKLists/OAK2trainDialogsAudios.txt  /local/1/anindita.nath/pitchCacheOAKDialogs8020/ /local/1/anindita.nath/OAK2trainFeatsAndLabels/ /research2/leo.boytsov/data/oak1_2/WAV.all/ /research2/leo.boytsov/pipelines/diarTrain2019-07-03/outStm/ 0 < /dev/null >& OAK2LFBEtrainSavePerAudioFeatsLabelsNoPitchPy_Aug13.out &

For 126 dimensions (same 6 low-vel features as above but now extracted from 10 Pastframes and 10 Future frames corresponding to each Present Frame features): 
nohup python cmdLnMainPastFuturePerAudio.py /research2/anindita.nath/OAKLists/OAK2trainDialogsAudios.txt  /local/1/anindita.nath/pitchCacheOAKDialogs8020/ /local/1/anindita.nath/OAK2trainFeatsAndLabels/ /research2/leo.boytsov/data/oak1_2/WAV.all/ /research2/leo.boytsov/pipelines/diarTrain2019-07-03/outStm/ 0 < /dev/null >& OAK2PastFuturetrainSavePerAudioFeatsLabelsNoPitchPy_Aug13.out &


<br> ***Aim: To train the networks.***

<br> **Run from ssh prompt**:

nohup python CPUflag start_core core_counts --use-devs --use-inter --use-intra --no-const-fold pathToTrainData pathToDevData pathToTestData pathToSaveModel typeOfModel saveToDisk trainShuffleSize nShuffleTrainBatches devShuffleSize testShuffleSize nHidden changeLabels deleteUnknowns batchSize learnRate epochs

<br>**Input**: 
CPUflag - 'CPU or GPU' (a string determining whether to run on CPU or GPU)
start_core - integer, serial number of the first core
core_counts - integer, total number of cores available or want to run on.
--use-devs - Use all devices if true
--use-inter - Use inter-device parallelism if true
--use-intra - Use intra-thread parallelism if true
--no-const-fold - Parallelism doesnot work if this is true, default is false.

pathToTrainData - 'Path to entire train set of .mat files
pathToDevData - Path to entire dev set of .mat files
pathToTestData - Path to entire test .mat files
pathToSaveModel - Path to the directory to where the generated model, graphs and results to be saved
typeOfModel - Either 'BiLSTM' or 'LSTM', string mentioning the deep learning algo to be used
nFeatures - integer, 6  for low-level, 64 for LFBE and  126 for past and future frames

saveToDisk - bool, if true, shuffled data is first saved to disk as 1 .mat file then loaded as needed batchwise.
trainShuffleSize - integer, number of files loaded and/or saved to disk from train set in a single iteration.
nShuffleTrainBatches- integer, number of iterations of training
devShuffleSize - integer,number of files loaded from dev set, if 0, then entire set is loaded.
testShuffleSize - integer, number of files loaded from test set,if 0, then entire set is loaded
nHidden - integer, number of hidden units in each layer of the network
changeLabels - bool, if True, the labels change from 0, 1,2 to -1,0 and 1
deleteUnknows - bool, if True, the unknown  labels (-1) along with their corresponding features are deleted from the respective dialog/audio set else they are masked during optimization.
batchSize - integer, number of files in a single batch of training
learnRate - float, initial learning rate and also the rate for L2 loss regularization.
epochs - integer, number of training epochs.

<br>**Sample Run**:-
Run from /../Codes:
nohup python end-to-endProsodyTurnCmdLn.py GPU 0 4 --use-devs --use-inter --use-intra --no-const-fold /local/1/anindita.nath/OAK2trainFeatsAndLabels/6dims/ /local/1/anindita.nath/OAK2devFeatsAndLabels/6dims/ /local/1/anindita.nath/testFeatsAndLabels/6dims/ /local/1/anindita.nath/OAK2trainFeatsAndLabels/Models/6dims/ LSTM 6 True 3 1 1 1 30 True True 1 0.001 1 </dev/null >& OAK2_6dimsNoMaskShuffleAug22.out &
