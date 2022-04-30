# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:40:13 2022

@author: Lin
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:31:41 2022

@author: Lin
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:08:40 2022

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

def bootstrapping_result(iterations, x_test, y_test, model_type):
    
    result_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                      'Accuracy','F1Score', 'AUC'])
    
    for i in range(iterations):
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
        # print("prediction shape:", prediction.shape)
        pred_result, auc = eva.ROC_plot(label = y_test, prediction = prediction, 
                               title= model_type, save_pic = False, path = str(model_type) + '.JPEG')
        
        table = confusion_matrix(y_test, pred_result)
        table_df = eva.result_original_matrix(table, auc_score = auc)
        # table_df
        
        frames = [result_df, table_df]
        result_df = pd.concat(frames)
    
    return result_df

#Random_Forest Result
boot_result = bootstrapping_result(iterations = 2, x_test = test_features, y_test = test_labels, model_type = 'Random_Forest')
boot_result
path = 'Random_Forest_boot_result.csv'
boot_result.to_csv(path, index = False)

#SVM Result
boot_result = bootstrapping_result(iterations = 2, x_test = test_features, y_test = test_labels, model_type = 'SVM')
boot_result
path = 'SVM_boot_result.csv'
boot_result.to_csv(path, index = False)

#DNN Result
boot_result = bootstrapping_result(iterations = 2, x_test = test_features, y_test = test_labels, model_type = 'DNN')
boot_result
path = 'DNN_boot_result.csv'
boot_result.to_csv(path, index = False)

#RNN Result
boot_result = bootstrapping_result(iterations = 2, x_test = test_features, y_test = test_labels, model_type = 'RNN')
boot_result
path = 'RNN_boot_result.csv'
boot_result.to_csv(path, index = False)

#Logistic Result
boot_result = bootstrapping_result(iterations = 2, x_test = test_features, y_test = test_labels, model_type = 'Logistic')
boot_result
path = 'Logistic_boot_result.csv'
boot_result.to_csv(path, index = False)