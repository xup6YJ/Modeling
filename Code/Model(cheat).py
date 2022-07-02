# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 01:33:28 2022

@author: Lin
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import Evaluation as eva
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, SimpleRNN
from tensorflow.keras import models, layers, optimizers
from keras.optimizers import Adam, SGD
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#Modeling

def dnn_model(train_input_shape):
    
    model = Sequential()
    model.add(Dense(units=512, activation='relu', input_shape = train_input_shape))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # model.summary()

    model.compile(loss='binary_crossentropy', 
                  optimizer  = optimizers.Adam (learning_rate = 0.001, name = 'adam'), metrics = ['acc'])
    
    return model

def rnn_model(train_input_shape):
    
    model = Sequential()
    model.add(SimpleRNN(units = 256, input_shape = train_input_shape, activation = 'relu'))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units = 16, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    # model.summary()
    
    
    model.compile(loss='binary_crossentropy', 
                  optimizer  = optimizers.Adam (learning_rate = 0.001, name = 'adam'), metrics = ['acc'])
    
    return model

def rf_model():
    
    # model = RandomForestRegressor(n_estimators = 10, criterion='squared_error', bootstrap = False, min_samples_split = 2,
    #                               min_samples_leaf = 10)  #can modify the number of trees 

    model = RandomForestRegressor(n_estimators=10, criterion='poisson', max_depth=None, min_samples_split=2, 
            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, 
            max_leaf_nodes=None, min_impurity_decrease=0.0, 
            bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=0.45)  

    return model

def svm_model():
    
    model = SVC(kernel='linear', max_iter = 35)
    return model

def logistic_reg():
    
    model = LogisticRegression(max_iter=10)
    return model


    
    
