# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:09:10 2022

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf 
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, SimpleRNN
from tensorflow.keras import models, layers, optimizers
from keras.optimizers import Adam, SGD
import seaborn as sn
import eli5
from eli5.sklearn import PermutationImportance

# importance = model.coef_
features = pd.read_csv('2018-2020.csv')  #convert path
labels = np.array(features['ADL-group'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('ADL-group', axis = 1)
features= features.drop('ID', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)

def dnn_model():
    
    model = Sequential()
    model.add(Dense(units=512, activation='relu', input_dim=20))  #convert to your own input_dim
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

model = dnn_model()

clf = KerasRegressor(build_fn=dnn_model, epochs=10, batch_size=64)
clf.fit(features, labels)

perm = PermutationImportance(clf, random_state=1).fit(features,labels)
w = eli5.show_weights(perm, feature_names = features.columns.tolist())

result = pd.read_html(w.data)[0]
result

# results = permutation_importance(model, X_train, y_train, scoring='neg_mean_squared_error')
