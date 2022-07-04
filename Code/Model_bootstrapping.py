
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
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as stats
from sklearn.naive_bayes import GaussianNB

import Evaluation as eva
import Model as md
from DataPreprocessing2 import *

# main

# data
data = dat('internal.csv')  #internal data

# main

# data

exv_x, exv_y = read_external_data()

#iteration = 1000
def bootstrapping(model_type, x_exv = exv_x, y_exv = exv_y, iterations = 2):
    
    result_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                      'Accuracy','F1Score', 'AUC'])
    exv_result_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                  'Accuracy','F1Score', 'AUC'])
    
    for i in range(iterations):
        
        print('No. ', i, ' iteration')
        
        #data shuffling
        train_features, test_features, train_labels, test_labels = data.shuffle_dat()
        x_test = test_features
        y_test = test_labels
        
        
        if model_type == 'DNN':
            train_shape = (train_features.shape[1],)
            model = md.dnn_model(train_shape)
            history  = model.fit(train_features, 
                                 train_labels, 
                                 epochs = 10, batch_size = 16)  #paper 70 epoch
            
        elif model_type == 'RNN':
            train_shape = (rnn_train_features.shape[1], rnn_train_features.shape[2])
            model = md.rnn_model(train_shape)
            history  = model.fit(rnn_train_features, 
                                 train_labels, 
                                 epochs = 10, batch_size = 16)  #paper 20 epoch
            
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
            history = model.fit(train_features, train_labels.ravel())
            
        elif model_type == 'Logistic':
            model = md.logistic_reg()
            model.fit(train_features,train_labels.ravel())
            
        elif model_type == 'KNN':
            model = KNeighborsClassifier(n_neighbors=5)
            history = model.fit(train_features,train_labels.ravel())
            
        elif model_type == 'NBC':
            model = GaussianNB()
            history = model.fit(train_features,train_labels.ravel())
            
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


def anova_boot(all_result_df, number_of_model):
    indexs = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy','F1Score', 'AUC']
    anova_matrix = x = np.zeros((number_of_model+1, 15))
    
    for i in range(number_of_model+1):
        
        if i <number_of_model :
            anova_matrix[i, 0] = np.mean(all_result_df['Sensitivity'][all_result_df['Group'] == i+1])
            anova_matrix[i, 1] = np.std(all_result_df['Sensitivity'][all_result_df['Group'] == i+1])
            
            anova_matrix[i, 2] = np.mean(all_result_df['Specificity'][all_result_df['Group'] == i+1])
            anova_matrix[i, 3] = np.std(all_result_df['Specificity'][all_result_df['Group'] == i+1])
            
            anova_matrix[i, 4] = np.mean(all_result_df['PPV'][all_result_df['Group'] == i+1])
            anova_matrix[i, 5] = np.std(all_result_df['PPV'][all_result_df['Group'] == i+1])
            
            anova_matrix[i, 6] = np.mean(all_result_df['NPV'][all_result_df['Group'] == i+1])
            anova_matrix[i, 7] = np.std(all_result_df['NPV'][all_result_df['Group'] == i+1])
            
            anova_matrix[i, 8] = np.mean(all_result_df['Accuracy'][all_result_df['Group'] == i+1])
            anova_matrix[i, 9] = np.std(all_result_df['Accuracy'][all_result_df['Group'] == i+1])
            
            anova_matrix[i, 10] = np.mean(all_result_df['F1Score'][all_result_df['Group'] == i+1])
            anova_matrix[i, 11] = np.std(all_result_df['F1Score'][all_result_df['Group'] == i+1])
            
            anova_matrix[i, 12] = np.mean(all_result_df['AUC'][all_result_df['Group'] == i+1])
            anova_matrix[i, 13] = np.std(all_result_df['AUC'][all_result_df['Group'] == i+1])
            
            anova_matrix[i, 14] = i+1
        
    anova_test = []
           
    for index in indexs:
         
         # you should delete or add code lines if your number of models is not 5
         # EX: you only need DNN, RNN, SVM then you should only reserve x1, x2, x3
         # Delete x4 = pd.to_numeric(all_result_df[index][all_result_df['Group'] == 4]) and
         # x5 = pd.to_numeric(all_result_df[index][all_result_df['Group'] == 5])
         # Change: a_result = stats.f_oneway(x1,x2,x3, x4, x5)   to    a_result = stats.f_oneway(x1,x2,x3)
         x1 = pd.to_numeric(all_result_df[index][all_result_df['Group'] == 1])
         x2 = pd.to_numeric(all_result_df[index][all_result_df['Group'] == 2])
         x3 = pd.to_numeric(all_result_df[index][all_result_df['Group'] == 3])
         x4 = pd.to_numeric(all_result_df[index][all_result_df['Group'] == 4])
         x5 = pd.to_numeric(all_result_df[index][all_result_df['Group'] == 5])
         #
         a_result = stats.f_oneway(x1, x2, x3, x4, x5)
         

         a_result = round(a_result[1],5)
         anova_test.append(a_result)
        
    for i, j in enumerate(range(0, 14, 2)):
        # print(i, j)
        anova_matrix[number_of_model, j] = anova_test[i]
            
    anova_df = pd.DataFrame(anova_matrix, columns=['Sensitivity Mean', 'Sensitivity SD',
                                              'Specificity Mean', 'Specificity SD',
                                              'PPV Mean', 'PPV SD',
                                              'NPV Mean', 'NPV SD', 
                                              'Accuracy Mean', 'Accuracy SD',
                                              'F1Score Mean', 'F1Score SD', 
                                              'AUC Mean', 'AUC SD', 
                                              'Group'])
    
    return anova_df

#Main
#all of the support model types on this code
all_model_type = ('DNN', 'RNN', 'Random Forest', 'SVM', 'Logistic', 'KNN', 'NBC')

#choose the models you want here (need >1)
sel_model_type = ('Random Forest', 'SVM', 'Logistic', 'NBC')

def boot_result(sel_model_type = sel_model_type):

    all_result_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                      'Accuracy','F1Score', 'AUC', 'Group', 'Model_Type'])
    
    all_eva_result_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                      'Accuracy','F1Score', 'AUC', 'Group', 'Model_Type'])
     
    
    if len(sel_model_type)>1:
        
        for i, mod in enumerate(sel_model_type):
            if mod in all_model_type:
                
                #Result
                boot_result, boot_eva_result = bootstrapping(model_type = mod)
                # path = 'Random_Forest_boot_result.csv'
                # boot_result.to_csv(path, index = False)
                # path = 'Random_Forest_boot_eva_result.csv'
                # boot_eva_result.to_csv(path, index = False)

                boot_result['Group'] = i+1
                boot_eva_result['Group'] = i+1
                boot_result['Model_Type'] = mod
                boot_eva_result['Model_Type'] = mod
                
                all_result_df = pd.concat([boot_result, all_result_df])
                all_eva_result_df = pd.concat([boot_result, all_eva_result_df])
        
        
        path = 'boot_result/all_result_df.csv'
        all_result_df.to_csv(path, index = False)
        path = 'boot_result/all_eva_result_df.csv'
        all_eva_result_df.to_csv(path, index = False)   
        
        anova_df = anova_boot(all_result_df, len(sel_model_type))
        anova_eva_df = anova_boot(all_eva_result_df, len(sel_model_type))

        path = 'boot_result/anova_df.csv'
        anova_df.to_csv(path, index = False)
        path = 'boot_result/anova_eva_df.csv'
        anova_eva_df.to_csv(path, index = False)


boot_result() 

