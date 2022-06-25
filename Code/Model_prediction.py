# -*- coding: utf-8 -*-
"""
Created on Mon May 30 2022

@author: Admin
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import Evaluation as eva
import Model as md
import DataPreprocessing as dp


# data
train_features, train_labels, test_features, test_labels = dp.read_features()
exv_x, exv_y = dp.read_external_data()

# fit the keras model on the dataset   
# Rnn feature
rnn_train_features = np.array(train_features)
rnn_train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1], 1))
rnn_train_features.shape

def model_prediction(x_test, y_test, model_type, x_exv = exv_x, y_exv = exv_y):
    
    if model_type == 'DNN':
        train_shape = (train_features.shape[1],)
        model = md.dnn_model(train_shape)
        history  = model.fit(train_features, 
                             train_labels, 
                             epochs = 10, batch_size = 16)  #70  #can convert 10/20
        
    elif model_type == 'RNN':
        train_shape = (rnn_train_features.shape[1], rnn_train_features.shape[2])
        model = md.rnn_model(train_shape)
        history  = model.fit(rnn_train_features, 
                             train_labels, 
                             epochs = 10, batch_size = 16)  #20  #can convert 10/20
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        x_exv = np.array(x_exv)
        x_exv = np.reshape(x_exv, (x_exv.shape[0], x_exv.shape[1], 1))
        
    elif model_type == 'Random Forest':
        # Instantiate model with 1000 decision trees, 50 node size
        model = md.rf_model()
        history = model.fit(train_features, train_labels.ravel())
    
    elif model_type == 'SVM':
        model = md.svm_model()
        history = model.fit(train_features,train_labels.ravel())
        
    elif model_type == 'Logistic':
        model = md.logistic_reg()
        model.fit(train_features,train_labels.ravel())
        
    elif model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=3)
        history = model.fit(train_features,train_labels.ravel())
        
    elif model_type == 'NBC':
        model = GaussianNB()
        history = model.fit(train_features,train_labels.ravel())
        
    # prediction
    prediction = model.predict(x_test)
    pred_result, auc = eva.ROC_plot(label = y_test, prediction = prediction, 
                                title= model_type, save_pic = False, path = str(model_type) + '.JPEG')
    
    table = confusion_matrix(y_test, pred_result)
    table_df = eva.result_original_matrix(table, auc_score = auc)
    
    # external prediction
    eva_prediction = model.predict(x_exv)
    pred_result, auc = eva.ROC_plot(label = y_exv, prediction = eva_prediction, 
                           title= model_type, save_pic = False, path = str(model_type) + '.JPEG')
    
    table = confusion_matrix(y_exv, pred_result)
    eva_table_df = eva.result_original_matrix(table, auc_score = auc)
    
    return [prediction, table_df, eva_prediction, eva_table_df]


#Main
#all of the model type reserved on this code
all_model_type = ('DNN', 'RNN', 'Random Forest', 'SVM', 'Logistic', 'KNN', 'NBC')

#choose the model you want here
sel_model_type = ('Random Forest', 'SVM', 'Logistic', 'KNN', 'NBC')


def all_result(sel_model_type = sel_model_type):

    all_pred_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                      'Accuracy','F1Score', 'AUC', 'Group', 'Model_Type'])
    
    all_eva_pred_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                      'Accuracy','F1Score', 'AUC', 'Group', 'Model_Type'])
            
    roc_color = ('blue', 'red', 'green', 'black', 'orange', 'purple', 'yellow')       
    
    
    fig1, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 120)
    plt.title('Internal Validation')
    
    fig2, ax2 = plt.subplots(1,1, figsize = (5, 5), dpi = 120)
    plt.title('External Validation')
    
    if len(sel_model_type)>1:
        
        for i, mod in enumerate(sel_model_type):
            if mod in all_model_type:
                
                #Result
                result = model_prediction(x_test = test_features, y_test = test_labels, model_type = mod)
                df = result[1].iloc[:,:4]
                path = 'cm/' + mod + '_cm.jpeg'
                title = mod +' Internal Validation'
                eva.plot_confusion(title, df = df, path = path)
        
                df = result[3].iloc[:,:4]
                path = 'cm/' + mod + 'eva_cm.jpeg'
                title = mod +' External Validation'
                eva.plot_confusion(title, df = df, path = path)
        
                result[1]['Group'] = i+1
                result[3]['Group'] = i+1
                result[1]['Model_Type'] = mod
                result[3]['Model_Type'] = mod
                all_pred_df = pd.concat([result[1], all_pred_df])
                all_eva_pred_df = pd.concat([result[3], all_eva_pred_df])
                
                path = 'result/all_pred_df.csv'
                all_pred_df.to_csv(path, index = False)
                path = 'result/all_eva_pred_df.csv'
                all_eva_pred_df.to_csv(path, index = False)
                
                #Internal
                fpr, tpr, thresholds  = roc_curve(test_labels, result[0])
                #Plot ROC and Save
                ax1.plot(fpr, tpr, 'b.-', label = mod + '(AUC:%2.2f)' % roc_auc_score(test_labels, result[0]) , color = roc_color[i])
        
                ax1.legend(loc = 4)
                ax1.set_xlabel('1 - Specificity')
                ax1.set_ylabel('Sensitivity')
                
                #External
                fpr2, tpr2, thresholds2  = roc_curve(exv_y, result[2])
                #Plot ROC and Save
                ax2.plot(fpr2, tpr2, 'b.-', label = mod + '(AUC:%2.2f)' % roc_auc_score(exv_y, result[2]), color = roc_color[i])
                
                ax2.legend(loc = 4)
                ax2.set_xlabel('1 - Specificity')
                ax2.set_ylabel('Sensitivity')
        
        fig1.savefig('result/ROC.jpeg')       
        fig2.savefig('result/ROC_eva.jpeg')        

all_result() 

