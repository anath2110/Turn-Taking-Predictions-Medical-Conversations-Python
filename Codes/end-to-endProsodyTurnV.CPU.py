# -*- coding: utf-8 -*-
from __future__ import print_function # since its always have to be at the beginning of the file
"""
Created on Tue Aug 13 11:41:40 2019

@author:Anindita Nath 
        3M|M*Modal
        Pittsburgh,PA  
"""
"""      
Reference:
Ward et al., "TURN-TAKING PREDICTIONS ACROSS LANGUAGES AND GENRES USING AN LSTM
RECURRENT NEURAL NETWORK", IEEE Workshop on Spoken Language Technology (SLT),2018,
10.1109/SLT.2018.8639673
"""
"""
Function that implements an end-to-end deep learning prosody based model
(LSTM or BiLSTM)to predict turns for any given speech/dialogs set.

"""

import sys
import tensorflow as tf
from loadStackDataNoSplit import loadStack
from loadStackDataNoSplit import loadShuffleSave
from loadStackDataNoSplit import loadFromShuffledStorage
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import os
from filesWithExtension import filesWithExtension
import random
from ftsLblstFformat import tfFormat
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Path To Stored Data Files
pathToTrainData='D:/FUNDataCodes/FeatsLabels/sampleOAK3Dialogs/\
1FutureFrame/PastFuture126dims/'
pathToDevData='D:/FUNDataCodes/FeatsLabels/sampleOAK3Dialogs/\
1FutureFrame/PastFuture126dims/'
pathToTestData='D:/FUNDataCodes/FeatsLabels/sampleOAK3Dialogs/\
1FutureFrame/PastFuture126dims/'
pathToSaveModel='D:/FUNDataCodes/ModelOPs/'

typeOfModel='LSTM' #or Bi-LSTM
# Data Dimension
nFeatures = 126   #Prosody, only 6 low-level, 
                #change this to 126 when features generated from 
                #featuresPastPresentFuturePerAudio.py    
                #change this to 64 when features generated from 
                #featuresLFBEAndLabelsNoSplitPerAudio.py   
timesteps = 1200 # 12sec input(each frame being 10ms) = 1200
nHidden = 30  # num of hidden units, also units in LSTM cell
keepProbTrain = 0.75
nClasses=1 # since there is only 1 column in label vectors with -1/0,0/1,1/2 labels



# Hyper-parameters
learnRate = 0.001  # The optimization initial learning rate 
                        # learn-rate sufficiently decresed to avoid 'nan' loss                                    
#trainSteps = 1200 #epochs
trainSteps = 5 #epochs
batchSize = 1    # Training batch size
displayStep = 1    # Frequency of displaying the training results
beta = 0.001



shuffleSize=3#dev set and test set are called with all the files each time
changeLabels=True # Booalean variable. 
                  # True means assign 0 as -1, 1as 0 and 2 as 1. 
                  # False means keep them as 0,1a nd 2.
deleteUnknows=True # Boolean variable. 
                   # True means delete the labels with either 0s or -1s 
                   # and their corresponding features from respective arrays.


#.............................................................................#


nIpUnits=nFeatures # number of input units

# Loading data
print("Load data...")

"""Shuffle and Store Validation Data"""
#Get the Validation Mat File Names
devFileNames=filesWithExtension(pathToDevData,'Lbls.mat')
nDevFiles=len(devFileNames)
print('validationdata')

pathToshuffledDevSet=loadShuffleSave(pathToDevData,typeOfModel,devFileNames,'Entire',
                                 nDevFiles,nIpUnits,'Dev',changeLabels,deleteUnknows)
print(pathToshuffledDevSet)

"""Load Validation Data from Shuffled Storage"""
validFeats, validLabels= loadFromShuffledStorage\
(pathToshuffledDevSet)
print('valid data Before formatting')
print(validFeats.shape)
print(validLabels.shape)

#transform feature arrays to conform to shape(batchsize,timesteps,nIpUnits)
validFeats,validLabels=tfFormat(validFeats,validLabels,timesteps,nClasses)
print('Valid data After formatting')
print(validFeats.shape)
print(validLabels.shape)

#.............................................................................#

"""Shuffle and Store Test Data"""
#Get all the Test .mat File Names
testFileNames=filesWithExtension(pathToTestData,'Lbls.mat')
nTestFiles=len(testFileNames)
print('test data')

pathToshuffledTestSet=loadShuffleSave(pathToTestData,typeOfModel,testFileNames,'Entire',
                                 nTestFiles,nIpUnits,'Test',changeLabels,deleteUnknows)
print(pathToshuffledTestSet)

"""Load Test Data from Shuffled Storage"""
testFeats, testLabels= loadFromShuffledStorage\
(pathToshuffledTestSet)
print('test data Before formatting')
print(testFeats.shape)
print(testLabels.shape)

#transform feature arrays to conform to shape(batchsize,timesteps,nIpUnits)
testFeats,testLabels=tfFormat(testFeats,testLabels,timesteps,nClasses)
print('test data After formatting')
print(testFeats.shape)
print(testLabels.shape)

#.............................................................................#

"""Shuffle and Store Train Data"""
#get all the Trainining .mat Files
trainFileNames=filesWithExtension(pathToTrainData,'Lbls.mat')
nTrainFiles=len(trainFileNames)

# shuffle the entire train set of mat files once
#print(trainFileNames)
random.shuffle(trainFileNames) 
#print(trainFileNames)

nShuffleTrainBatches=int(nTrainFiles/shuffleSize)
#print(nShuffleTrainBatches)
for nthShuffledTrainBatch in range(nShuffleTrainBatches):
    pathToshuffledTrainSet=loadShuffleSave(pathToTrainData,typeOfModel,trainFileNames,
                            nthShuffledTrainBatch,nTrainFiles,nIpUnits,'Train',
                                                   changeLabels,deleteUnknows)
    #print(pathToshuffledTrainSet)
    trainFeats, trainLabels = loadFromShuffledStorage(pathToshuffledTrainSet)
    print('Train data Before formatting')
    print(trainFeats.shape)
    print(trainLabels.shape)
    
    #transform feature arrays to conform to shape(batchsize,timesteps,nIpUnits)
    trainFeats,trainLabels=tfFormat(trainFeats,trainLabels,timesteps,nClasses)
    print('Train data After formatting')
    print(trainFeats.shape)
    print(trainLabels.shape)
   
    """Start drawing the tensorflow graph"""
    tf.reset_default_graph()
    #print('in train data loop')
   
    # tf Graph input
    #Placeholders for inputs (x) and outputs(y)
    X = tf.placeholder("float", [None, timesteps,nIpUnits])
    Y = tf.placeholder("float", [None,timesteps,nClasses])
    
    keepProb = tf.placeholder(tf.float32)  # dropout (keep probability)
    
    # Define weights/biases
    if typeOfModel=='BiLSTM':
        weights = {
        'hidden1': tf.get_variable("w_hid1", shape=(nIpUnits, nIpUnits),
                                
                                 initializer=tf.contrib.layers.xavier_initializer()),
    
        'hidden2': tf.get_variable("w_hid2", shape=(nIpUnits, nIpUnits),
                                  
                                   initializer=tf.contrib.layers.xavier_initializer()),
    
        'out': tf.get_variable("w_out", shape=[nHidden*2, nClasses],
               initializer=tf.contrib.layers.xavier_initializer())
    }
    elif typeOfModel=='LSTM' :
        
        weights = {
            'hidden1': tf.get_variable("w_hid1", shape=(nIpUnits, nIpUnits),
                                    
                                     initializer=tf.contrib.layers.xavier_initializer()),
        
            'hidden2': tf.get_variable("w_hid2", shape=(nIpUnits, nIpUnits),
                                      
                                       initializer=tf.contrib.layers.xavier_initializer()),                                   
        
            'out': tf.get_variable("w_out", shape=[nHidden, nClasses],
                   initializer=tf.contrib.layers.xavier_initializer())
        }
    
    biases = {
        'hidden1': tf.get_variable("b_hid1", shape=[nIpUnits],
                               initializer=tf.contrib.layers.xavier_initializer()),
    
        'hidden2': tf.get_variable("b_hid2", shape=[nIpUnits],
                                  initializer=tf.contrib.layers.xavier_initializer()),
    
        'out': tf.get_variable("b_out", shape=[nClasses],
               initializer=tf.contrib.layers.xavier_initializer())
    }
    
    
    
    def parametricRelu(_x, name):
        alpha = tf.get_variable(name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.1),
                                 dtype=_x.dtype)
        pos = tf.nn.relu(_x)
        neg = alpha * (_x - abs(_x)) * 0.5
    
        return pos + neg
    
    
    def createModel(x, weights, biases):
    
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batchSize, timesteps,nIpUnits)
        # Required structure: list of size 'timesteps',
        # where each item in the list is a tensor of shape: (batchSize, nIpUnits)        
    
        x = tf.reshape(x, [-1,nIpUnits])      
        #print(x.shape)
        #1st Hidden Prelu Layer
        x = tf.nn.bias_add(tf.matmul(x, weights['hidden1']), biases['hidden1'])  
        x = parametricRelu(x, "alpha_h1")
        x = tf.nn.dropout(x, keepProb)
        
        #2nd Hidden Prelu Layer
        x = tf.nn.bias_add(tf.matmul(x, weights['hidden2']), biases['hidden2'])  
        x = parametricRelu(x, "alpha_h2")
        x = tf.nn.dropout(x, keepProb)
    
        x = tf.reshape(x, [-1, timesteps,nIpUnits])        
        # Unstack  to get a list of 'timesteps' tensors of shape (batchSize, nIpUnits)
        x = tf.unstack(x, timesteps, 1)
        #print(x[0].shape)
        
        if typeOfModel=='BiLSTM':
            # Define lstm cells with tensorflow
            # Forward direction cell
            lstm_fw_cell = rnn.BasicLSTMCell(nHidden, forget_bias=1.0)
            # Backward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(nHidden, forget_bias=1.0)
            
            
            # Get BiRNN cell output
            try:
                outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
            except Exception: # Old TensorFlow version only returns outputs not states
                outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)  
            
           # outputs is a list of tensors of shape (2*batchSize, nHidden).
            # Size of the list: timesteps
           
            #print(outputs[0].shape)
            outputs = tf.stack(outputs, axis=1)
            # Reshape 'outputs' to be a 2D matrix, so we can perform outputs * weights
            #outputs = tf.reshape(outputs, [-1, nHidden])
            outputs = tf.reshape(outputs, [-1, nHidden*2])
            
        elif typeOfModel=='LSTM' :
    
            # Basic LSTM Cell with nHidden units
            lstm_cell = rnn.BasicLSTMCell(nHidden, forget_bias=1.0)        
            #static lstm rnn 
            # outputs is a list of tensors of shape (batchSize, nHidden).
            # Size of the list: timesteps
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
            
            #print(outputs[0].shape)
            # Reshape 'outputs' to be a 2D matrix, so we can perform outputs * weights
            outputs = tf.reshape(outputs, [-1, nHidden])
       
        outputsWB=tf.add(tf.matmul(outputs, weights['out']),biases['out'])
       
        # Reshape the result back to (timesteps, batchSize, nClasses)
        outputsNonSig = tf.reshape(outputsWB, [timesteps, -1, nClasses])    
        # The shape of the prediction must be (batchSize, timesteps,nClasses)
        outputsNonSig = tf.transpose(outputsNonSig, [1,0,2])
        #print(y_pred.shape)
        
        
        #y_pred = tf.nn.softmax(outputsWB) # no need loss function takes care of it
        y_pred = tf.nn.sigmoid(outputsWB) 
        # Reshape the result back to (timesteps, batchSize, nClasses)
        y_pred = tf.reshape(y_pred, [timesteps, -1, nClasses])    
        # The shape of the prediction must be (batchSize, timesteps,nClasses)
        y_pred = tf.transpose(y_pred, [1,0,2])
        #print(y_pred.shape)
        return y_pred,outputsNonSig
    
    print("Building graph...")
    networkOp,outputsNonSig = createModel(X, weights, biases)
    
    if deleteUnknows==False:
        networkOp = tf.reshape(networkOp, [-1])
        outputsNonSig = tf.reshape(outputsNonSig, [-1])
        labels = tf.reshape(Y, [-1])
        index = tf.where(tf.not_equal(labels, tf.constant(-1, dtype=tf.float32)))        
        labels = tf.gather(labels, index)
        networkOp = tf.gather(networkOp, index)
        outputsNonSig = tf.gather(outputsNonSig, index)
        
        lossOp = \
        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                   (labels=labels, logits=outputsNonSig))  
        accuracy,acc_op=tf.metrics.accuracy(labels, networkOp)
    elif deleteUnknows==True:
        lossOp = \
        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                   (labels=Y, logits=outputsNonSig))  
        accuracy,acc_op=tf.metrics.accuracy(Y, networkOp)

#        print(outputsNonSig.shape)
#        print(Y.shape)
    """Define loss and optimizer""" 
    
#    lossOp = \
#    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2
#                   (labels=Y, logits=networkOp))  
    
    
    
    
    regularizer = tf.nn.l2_loss(weights['out'])
    lossOp = tf.reduce_mean(lossOp + beta * regularizer) 
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learnRate)
    trainOp = optimizer.minimize(lossOp)

    
    
    # Initialize the variables (i.e. assign their default value)
    localint=tf.local_variables_initializer()    
    init = tf.global_variables_initializer()
    
    print("Starting session... ")
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    """Train"""
    # Start training
    with tf.Session() as sess:       
        # Run the initializer
        sess.run(init)
        sess.run(localint)
        nbatches = int(trainFeats.shape[0] / batchSize)
        epochs=[]
        valLoss=[]
        for epoch in range(trainSteps):
            for step in range(nbatches):
                idx = np.random.randint(trainFeats.shape[0], size=batchSize)
                batch_x = trainFeats[idx, :, :]
                #if trainLabels[idx] !=0: #only if it is non-zero label
                batch_y = trainLabels[idx] 
    
                # Run optimization (backprop)
                _,_,train_loss=sess.run([networkOp,trainOp,lossOp], 
                                        feed_dict={X: batch_x, Y: batch_y,
                                                   keepProb: keepProbTrain})
#                
#                print('Train Loss at step {}: {}'.format(step, train_loss))
#    
#            
          
            _,validation_loss=sess.run\
            ([networkOp,lossOp], feed_dict={X: validFeats, 
             Y: validLabels, keepProb: 1.0})
    
            # epoch_cost /= num_batches          
          
            print("Epoch " + str(epoch) + ", Validation Loss= " + 
                  str(validation_loss))
            
            """ Save History"""
            epochs.append(epoch)            
            valLoss.append(validation_loss)
            
        epochs=np.asarray(epochs)       
        valLoss=np.asarray(valLoss)
            
        print("Optimization Finished!")     
    
        """ Save Model"""  
        #modelname=os.path.basename(__file__) + str(nFeatures)
        modelname=typeOfModel + str(nFeatures)
        save_path = saver.save(sess, pathToSaveModel +
                    modelname + '_'  + str(nthShuffledTrainBatch) + ".ckpt")
        print("Model saved in file: %s" % save_path)
        
        """ Plot Graphs for Visualization"""
        # Create new directory
        
        graph_dir= pathToSaveModel + 'Graphs' + "_" + modelname + "_" + '/'
        if not os.path.isdir(graph_dir):
            os.makedirs(graph_dir)
        
        # Summarize history for loss
       
        figloss=plt.figure()
        plt.plot(epochs,valLoss)
        plt.title(modelname + '_' + str(nthShuffledTrainBatch) + '_Loss')
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch')
        plt.legend(['dev'], loc='upper right')
        plt.show()
        figloss.savefig(graph_dir + modelname + '_' + 
                        str(nthShuffledTrainBatch) +'_Loss' + '.png')
        
        """Test"""
        
        test_loss = \
        sess.run([lossOp],
                 feed_dict={X: testFeats, Y: testLabels, keepProb: 1.0})
        print("Test Loss= " + str(test_loss))
        