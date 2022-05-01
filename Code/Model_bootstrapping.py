# -*- coding: utf-8 -*-


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
exv_x, exv_y = dp.read_external_data()

# fit the keras model on the dataset   
# Rnn feature
rnn_train_features = np.array(train_features)
rnn_train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1], 1))
rnn_train_features.shape

def bootstrapping_result(iterations, model_type, x_test = test_features, y_test = test_labels, x_exv = exv_x, y_exv = exv_y):
    
    result_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                      'Accuracy','F1Score', 'AUC'])
    exv_result_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                  'Accuracy','F1Score', 'AUC'])
    
    for i in range(iterations):
        if model_type == 'DNN':
            train_shape = (train_features.shape[1],)
            model = md.dnn_model(train_shape)
            history  = model.fit(train_features, 
                                 train_labels, 
                                 epochs = 1, batch_size = 16)  #paper 70 epoch
            
        elif model_type == 'RNN':
            train_shape = (rnn_train_features.shape[1], rnn_train_features.shape[2])
            model = md.rnn_model(train_shape)
            history  = model.fit(rnn_train_features, 
                                 train_labels, 
                                 epochs = 1, batch_size = 16)  #paper 20 epoch
            
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            x_exv = np.array(x_exv)
            x_exv = np.reshape(x_exv, (x_exv.shape[0], x_exv.shape[1], 1))
            
        elif model_type == 'Random_Forest':
            # Instantiate model with 1000 decision trees, 50 node size
            model = md.rf_model()
            history = model.fit(train_features, train_labels.ravel())
        
        elif model_type == 'SVM':
            model = md.svm_model()
            history = model.fit(train_features, train_labels.ravel())
            
        elif model_type == 'Logistic':
            model = md.logistic_reg()
            model.fit(train_features,train_labels.ravel())
            
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
        
        # external prediction
        prediction = model.predict(x_exv)
        # print("prediction shape:", prediction.shape)
        pred_result, auc = eva.ROC_plot(label = y_exv, prediction = prediction, 
                               title= model_type, save_pic = False, path = str(model_type) + '.JPEG')
        
        table = confusion_matrix(y_exv, pred_result)
        eva_table_df = eva.result_original_matrix(table, auc_score = auc)
        # table_df
        
        frames = [exv_result_df, eva_table_df]
        exv_result_df = pd.concat(frames)
        
    
    return result_df, exv_result_df

# iteration = 1000
#Random_Forest Result
boot_result, boot_eva_result = bootstrapping_result(iterations = 2, model_type = 'Random_Forest')
path = 'Random_Forest_boot_result.csv'
boot_result.to_csv(path, index = False)
path = 'Random_Forest_boot_eva_result.csv'
boot_eva_result.to_csv(path, index = False)

#SVM Result
boot_result, boot_eva_result = bootstrapping_result(iterations = 2, model_type = 'SVM')
path = 'SVM_boot_result.csv'
boot_result.to_csv(path, index = False)
path = 'SVM_boot_eva_result.csv'
boot_eva_result.to_csv(path, index = False)

#DNN Result
boot_result, boot_eva_result = bootstrapping_result(iterations = 2, model_type = 'DNN')
path = 'DNN_boot_result.csv'
boot_result.to_csv(path, index = False)
path = 'DNN_boot_eva_result.csv'
boot_eva_result.to_csv(path, index = False)

#RNN Result
boot_result, boot_eva_result = bootstrapping_result(iterations = 2, model_type = 'RNN')
path = 'RNN_boot_result.csv'
boot_result.to_csv(path, index = False)
path = 'RNN_boot_eva_result.csv'
boot_eva_result.to_csv(path, index = False)

#Logistic Result
boot_result, boot_eva_result = bootstrapping_result(iterations = 2, model_type = 'Logistic')
path = 'Logistic_boot_result.csv'
boot_result.to_csv(path, index = False)
path = 'Logistic_boot_eva_result.csv'
boot_eva_result.to_csv(path, index = False)
