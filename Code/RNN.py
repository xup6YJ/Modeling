# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:54:27 2022

@author: Lin
"""

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, SimpleRNN
from tensorflow.keras import models, layers, optimizers
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
#model architecture


# feature dim = smaple, variable, 1
train_features.shape
rnn_train_features = np.array(train_features)
# np.argwhere(np.isnan(rnn_train_features))

rnn_train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1], 1))
rnn_train_features.shape

model = Sequential()
model.add(SimpleRNN(units = 256, 
                    input_shape = (rnn_train_features.shape[1], rnn_train_features.shape[2]), activation = 'relu'))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


model.compile(loss='binary_crossentropy', 
              optimizer  = optimizers.Adam (lr = 0.001, name = 'adam'), metrics = ['acc'])

# fit the keras model on the dataset
history  = model.fit(rnn_train_features, 
                     train_labels, 
                     epochs = 20, batch_size = 16)

# main
pred_result = ROC_plot(label = test_labels, prediction = prediction, title= 'RNN', path = 'RNN.JPEG')

table = confusion_matrix(test_labels, pred_result)
table_df = result_original_matrix(table)
table_df
    
path = 'DNN_result.csv'
table_df.to_csv(path, index = False)