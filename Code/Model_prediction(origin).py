# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 11:39:04 2022

@author: Admin
"""

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

# main

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
        
    elif model_type == 'Random_Forest':
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
all_pred_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                  'Accuracy','F1Score', 'AUC', 'Group'])

all_eva_pred_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                  'Accuracy','F1Score', 'AUC', 'Group'])

# 1: RF, 2: SVM, 3: Logistic, 4: KNN, 5: NBC

#Random_Forest Result
rf_result = model_prediction(x_test = test_features, y_test = test_labels, model_type = 'Random_Forest')
path = 'result/Random_Forest_result.csv'
# rf_result[1].to_csv(path, index = False)
df = rf_result[1].iloc[:,:4]
path = 'cm/Random_Forest_cm.jpeg'
eva.plot_confusion('RF Internal Validation', df = df, path = path)

path = 'result/Random_Forest_eva_result.csv'
# rf_result[3].to_csv(path, index = False)
df = rf_result[3].iloc[:,:4]
path = 'cm/Random_Forest_eva_cm.jpeg'
eva.plot_confusion('RF External Validation', df = df, path = path)

rf_result[1]['Group'] = 1
rf_result[3]['Group'] = 1
all_pred_df = pd.concat([rf_result[1], all_pred_df])
all_eva_pred_df = pd.concat([rf_result[3], all_eva_pred_df])


#SVM Result
svm_result = model_prediction(x_test = test_features, y_test = test_labels, model_type = 'SVM')
path = 'result/SVM_result.csv'
# svm_result[1].to_csv(path, index = False)
df = svm_result[1].iloc[:,:4]
path = 'cm/SVM_cm.jpeg'
eva.plot_confusion('SVM Internal Validation', df = df, path = path)

path = 'result/SVM_eva_result.csv'
# svm_result[3].to_csv(path, index = False)
df = svm_result[3].iloc[:,:4]
path = 'cm/SVM_eva_cm.jpeg'
eva.plot_confusion('SVM External Validation', df = df, path = path)

svm_result[1]['Group'] = 2
svm_result[3]['Group'] = 2
all_pred_df = pd.concat([svm_result[1], all_pred_df])
all_eva_pred_df = pd.concat([svm_result[3], all_eva_pred_df])

#Logistic Result
lg_result = model_prediction(x_test = test_features, y_test = test_labels, model_type = 'Logistic')
path = 'result/Logistic_result.csv'
# lg_result[1].to_csv(path, index = False)
df = lg_result[1].iloc[:,:4]
path = 'cm/Logistic_cm.jpeg'
eva.plot_confusion('Logistic Internal Validation', df = df, path = path)

path = 'result/Logistic_eva_result.csv'
# lg_result[3].to_csv(path, index = False)
df = lg_result[3].iloc[:,:4]
path = 'cm/Logistic_eva_cm.jpeg'
eva.plot_confusion('Logistic External Validation', df = df, path = path)

lg_result[1]['Group'] = 3
lg_result[3]['Group'] = 3
all_pred_df = pd.concat([lg_result[1], all_pred_df])
all_eva_pred_df = pd.concat([lg_result[3], all_eva_pred_df])

#KNN Result
knn_result = model_prediction(x_test = test_features, y_test = test_labels, model_type = 'KNN')
path = 'result/KNN_result.csv'
# knn_result[1].to_csv(path, index = False)
df = knn_result[1].iloc[:,:4]
path = 'cm/KNN_cm.jpeg'
eva.plot_confusion('KNN Internal Validation', df = df, path = path)

path = 'result/KNN_eva_result.csv'
# knn_result[3].to_csv(path, index = False)
df = knn_result[3].iloc[:,:4]
path = 'cm/KNN_eva_cm.jpeg'
eva.plot_confusion('KNN External Validation', df = df, path = path)

knn_result[1]['Group'] = 4
knn_result[3]['Group'] = 4
all_pred_df = pd.concat([knn_result[1], all_pred_df])
all_eva_pred_df = pd.concat([knn_result[3], all_eva_pred_df])

#NBC Result
nbc_result = model_prediction(x_test = test_features, y_test = test_labels, model_type = 'NBC')
path = 'result/NBC_result.csv'
# nbc_result[1].to_csv(path, index = False)
df = nbc_result[1].iloc[:,:4]
path = 'cm/NBC_cm.jpeg'
eva.plot_confusion('NBC Internal Validation', df = df, path = path)

path = 'result/NBC_eva_result.csv'
# nbc_result[3].to_csv(path, index = False)
df = nbc_result[3].iloc[:,:4]
path = 'cm/NBC_eva_cm.jpeg'
eva.plot_confusion('NBC External Validation', df = df, path = path)

nbc_result[1]['Group'] = 5
nbc_result[3]['Group'] = 5
all_pred_df = pd.concat([nbc_result[1], all_pred_df])
all_eva_pred_df = pd.concat([nbc_result[3], all_eva_pred_df])

path = 'result/all_pred_df.csv'
all_pred_df.to_csv(path, index = False)
path = 'result/all_eva_pred_df.csv'
all_eva_pred_df.to_csv(path, index = False)

#Plot 
# Random_Forest ROC
fpr, tpr, thresholds = roc_curve(test_labels, rf_result[0])
#SVM ROC
fpr2, tpr2, thresholds2  = roc_curve(test_labels, svm_result[0])
#DNN ROC
#fpr3, tpr3, thresholds3  = roc_curve(test_labels, dnn_result[0])
#RNN ROC
#fpr4, tpr4, thresholds4  = roc_curve(test_labels, rnn_result[0])
#Logistic ROC
fpr5, tpr5, thresholds5  = roc_curve(test_labels, lg_result[0])
#KNN ROC
fpr6, tpr6, thresholds6  = roc_curve(test_labels, knn_result[0])
#NBC ROC
fpr7, tpr7, thresholds6  = roc_curve(test_labels, nbc_result[0])

fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 120)

#Plot ROC and Save
ax1.plot(fpr, tpr, 'b.-', label = 'Random Forest (AUC:%2.2f)' % roc_auc_score(test_labels, rf_result[0]))
ax1.plot(fpr2, tpr2, 'b.-', label = 'SVM (AUC:%2.2f)' % roc_auc_score(test_labels, svm_result[0]), color = 'r')
#ax1.plot(fpr3, tpr3, 'b.-', label = 'DNN (AUC:%2.2f)' % roc_auc_score(test_labels, dnn_result[0]), color = 'g')
#ax1.plot(fpr4, tpr4, 'b.-', label = 'RNN (AUC:%2.2f)' % roc_auc_score(test_labels, rnn_result[0]), color = 'black')
ax1.plot(fpr5, tpr5, 'b.-', label = 'Logistic (AUC:%2.2f)' % roc_auc_score(test_labels, lg_result[0]), color = 'yellow')
ax1.plot(fpr6, tpr6, 'b.-', label = 'KNN (AUC:%2.2f)' % roc_auc_score(test_labels, knn_result[0]), color = 'orange')
ax1.plot(fpr7, tpr7, 'b.-', label = 'NBC (AUC:%2.2f)' % roc_auc_score(test_labels, nbc_result[0]), color = 'purple')


ax1.legend(loc = 4)
ax1.set_xlabel('1 - Specificity')
ax1.set_ylabel('Sensitivity')
plt.title('Internal Validation')
fig.savefig('result/ROC.jpeg')

#Eva Plot
# Random_Forest ROC
test_labels = exv_y
fpr, tpr, thresholds = roc_curve(test_labels, rf_result[2])
#SVM ROC
fpr2, tpr2, thresholds2  = roc_curve(test_labels, svm_result[2])
#DNN ROC
#fpr3, tpr3, thresholds3  = roc_curve(test_labels, dnn_result[2])
#RNN ROC
#fpr4, tpr4, thresholds4  = roc_curve(test_labels, rnn_result[2])
#Logistic ROC
fpr5, tpr5, thresholds5  = roc_curve(test_labels, lg_result[2])
#KNN ROC
fpr6, tpr6, thresholds6  = roc_curve(test_labels, knn_result[2])
#NBC ROC
fpr7, tpr7, thresholds6  = roc_curve(test_labels, nbc_result[2])

fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 120)

#Plot ROC and Save
ax1.plot(fpr, tpr, 'b.-', label = 'Random Forest (AUC:%2.2f)' % roc_auc_score(test_labels, rf_result[2]))
ax1.plot(fpr2, tpr2, 'b.-', label = 'SVM (AUC:%2.2f)' % roc_auc_score(test_labels, svm_result[2]), color = 'r')
#ax1.plot(fpr3, tpr3, 'b.-', label = 'DNN (AUC:%2.2f)' % roc_auc_score(test_labels, dnn_result[2]), color = 'g')
#ax1.plot(fpr4, tpr4, 'b.-', label = 'RNN (AUC:%2.2f)' % roc_auc_score(test_labels, rnn_result[2]), color = 'black')
ax1.plot(fpr5, tpr5, 'b.-', label = 'Logistic (AUC:%2.2f)' % roc_auc_score(test_labels, lg_result[2]), color = 'yellow')
ax1.plot(fpr6, tpr6, 'b.-', label = 'KNN (AUC:%2.2f)' % roc_auc_score(test_labels, knn_result[2]), color = 'orange')
ax1.plot(fpr7, tpr7, 'b.-', label = 'NBC (AUC:%2.2f)' % roc_auc_score(test_labels, nbc_result[2]), color = 'purple')

ax1.legend(loc = 4)
ax1.set_xlabel('1 - Specificity')
ax1.set_ylabel('Sensitivity')
plt.title('External Validation')
fig.savefig('result/ROC_eva.jpeg')

