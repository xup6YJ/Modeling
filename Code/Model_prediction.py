# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 19:34:24 2022

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import Evaluation as eva
import Model as md
import DataPreprocessing as dp

# main

# data
train_features, train_labels, test_features, test_labels = dp.read_features()

# fit the keras model on the dataset   
# Rnn feature
rnn_train_features = np.array(train_features)
rnn_train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1], 1))
rnn_train_features.shape

def model_prediction(x_test, y_test, model_type):
    
    if model_type == 'DNN':
        train_shape = (train_features.shape[1],)
        model = md.dnn_model(train_shape)
        history  = model.fit(train_features, 
                             train_labels, 
                             epochs = 1, batch_size = 16)
        
    elif model_type == 'RNN':
        train_shape = (rnn_train_features.shape[1], rnn_train_features.shape[2])
        model = md.rnn_model(train_shape)
        history  = model.fit(rnn_train_features, 
                             train_labels, 
                             epochs = 1, batch_size = 16)
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
    elif model_type == 'Random_Forest':
        # Instantiate model with 1000 decision trees, 50 node size
        model = md.rf_model()
        history = model.fit(train_features, train_labels)
    
    elif model_type == 'SVM':
        model = md.svm_model()
        history = model.fit(train_features,train_labels)
        
    elif model_type == 'Logistic':
        model = md.logistic_reg()
        model.fit(train_features,train_labels)
        
    # prediction
    prediction = model.predict(x_test)
    pred_result, auc = eva.ROC_plot(label = y_test, prediction = prediction, 
                                title= model_type, save_pic = False, path = str(model_type) + '.JPEG')
    
    table = confusion_matrix(y_test, pred_result)
    table_df = eva.result_original_matrix(table, auc_score = auc)
    
    return prediction, table_df


#Random_Forest Result
rf_pred, rf_result = model_prediction(x_test = test_features, y_test = test_labels, model_type = 'Random_Forest')
path = 'Random_Forest_result.csv'
rf_result.to_csv(path, index = False)

#SVM Result
svm_pred, svm_result = model_prediction(x_test = test_features, y_test = test_labels, model_type = 'SVM')
path = 'SVM_result.csv'
svm_result.to_csv(path, index = False)

#DNN Result
dnn_pred, dnn_result = model_prediction(x_test = test_features, y_test = test_labels, model_type = 'DNN')
path = 'DNN_result.csv'
dnn_result.to_csv(path, index = False)

#RNN Result
rnn_pred, rnn_result = model_prediction(x_test = test_features, y_test = test_labels, model_type = 'RNN')
path = 'RNN_result.csv'
rnn_result.to_csv(path, index = False)

#Logistic Result
lg_pred, lg_result = model_prediction(x_test = test_features, y_test = test_labels, model_type = 'Logistic')
path = 'Logistic_result.csv'
lg_result.to_csv(path, index = False)

#Plot 
# Random_Forest ROC
fpr, tpr, thresholds = roc_curve(test_labels, rf_pred)
#SVM ROC
fpr2, tpr2, thresholds2  = roc_curve(test_labels, svm_pred)
#DNN ROC
fpr3, tpr3, thresholds3  = roc_curve(test_labels, dnn_pred)
#RNN ROC
fpr4, tpr4, thresholds4  = roc_curve(test_labels, rnn_pred)
#Logistic ROC
fpr5, tpr5, thresholds5  = roc_curve(test_labels, lg_pred)

fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 120)

#Plot ROC and Save
ax1.plot(fpr, tpr, 'b.-', label = 'Random Forest (AUC:%2.2f)' % roc_auc_score(test_labels, rf_pred))
ax1.plot(fpr2, tpr2, 'b.-', label = 'SVM (AUC:%2.2f)' % roc_auc_score(test_labels, svm_pred), color = 'r')
ax1.plot(fpr3, tpr3, 'b.-', label = 'DNN (AUC:%2.2f)' % roc_auc_score(test_labels, dnn_pred), color = 'g')
ax1.plot(fpr4, tpr4, 'b.-', label = 'RNN (AUC:%2.2f)' % roc_auc_score(test_labels, rnn_pred), color = 'black')
ax1.plot(fpr5, tpr5, 'b.-', label = 'Logistic (AUC:%2.2f)' % roc_auc_score(test_labels, lg_pred), color = 'yellow')

ax1.legend(loc = 4)
ax1.set_xlabel('1 - Specificity')
ax1.set_ylabel('Sensitivity')
plt.title('Model Prediction')

