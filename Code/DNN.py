# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import Evaluation as eva
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

# Read data
features = pd.read_csv('sig_var.csv')
features.head(5)

# Checking data
print('The shape of our features is:', features.shape)
features.describe()
features.head()

# checking null
print(features.isnull().sum())

# Labeling, featuring
labels = np.array(features['ADL-group'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('ADL-group', axis = 1)
features= features.drop('ID', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# train 0.7, test 0.3
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

sum(test_labels)/len(test_features)
sum(train_labels)/len(train_features)


#model architecture
model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape = (train_features.shape[1], )))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', 
              optimizer  = optimizers.Adam (lr = 0.001, name = 'adam'), metrics = ['acc'])

# fit the keras model on the dataset
history  = model.fit(train_features, 
                     train_labels, 
                     epochs = 20, batch_size = 16)

prediction = model.predict(test_features)
print("prediction shape:", prediction.shape)

#Evaludation main
pred_result = eva.ROC_plot(label = test_labels, prediction = prediction, title= 'DNN', path = 'DNN.JPEG')

table = confusion_matrix(test_labels, pred_result)
table_df = eva.result_original_matrix(table)
table_df
    
path = 'DNN_result.csv'
table_df.to_csv(path, index = False)
