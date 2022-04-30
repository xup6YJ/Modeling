# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 19:17:48 2022

@author: Lin
"""
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
                  optimizer  = optimizers.Adam (lr = 0.001, name = 'adam'), metrics = ['acc'])
    
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
                  optimizer  = optimizers.Adam (lr = 0.001, name = 'adam'), metrics = ['acc'])
    
    return model

def rf_model():
    
    model = RandomForestRegressor(n_estimators = 1000, 
                                  min_samples_leaf = 50)  #can modify the number of trees
    return model

def svm_model():
    
    model = SVC(kernel='linear')
    return model

def logistic_reg():
    
    model = LogisticRegression()
    return model


    
    
