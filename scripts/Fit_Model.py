#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:35:38 2020

@author: jordanshimer
"""
import numpy as np
 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout

from keras import backend as K
import tensorflow as tf
import os

# I wanna go fast!
NUM_PARALLEL_EXEC_UNITS = 8
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                       allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.Session(config=config)
K.set_session(session)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

def load_data():
    X = np.load('../data/arrays/target.npy')
    y = np.load('../data/arrays/input.npy')
    source = np.load('../data/arrays/source.npy')
    
    return(X, y, source)
    
def fit_model(X, y, epochs):
    X = X.astype('int64')
    y = y.astype('int64')

    vocab_size = np.max([np.max(X), np.max(y)]) + 1
    batch_size = 64
        
    model = Sequential()
    model.add(Embedding(vocab_size, 8))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    # compile network
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    # fit network
    model.fit(X, y, epochs=epochs, verbose=2, batch_size = batch_size)
    model.save('../model/model')
    
    return(model)

if __name__ == '__main__':
    X, y, source = load_data()
    #X, y = oversample(X, y, source)
     
    model = fit_model(X, y, 40)
    
    
    