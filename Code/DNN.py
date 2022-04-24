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

#data
features = pd.read_csv('D:/File_X/Help/hsu/sig_var.csv')
features.head(5)

# 探索資料
print('The shape of our features is:', features.shape)
features.describe()
features.head()

# checking null
print(features.isnull().sum())

# 特徵轉換
labels = np.array(features['ADL-group'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('ADL-group', axis = 1)
features= features.drop('ID', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

#train 0.7, test 0.3
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

#Evaludation
def ROC_plot(label, prediction, title, path):
    
    #Plot
    fpr, tpr, thresholds = roc_curve(label, prediction)
    fig, ax1 = plt.subplots(1,1, figsize = (10, 10), dpi = 80)
    ax1.plot(fpr, tpr, 'b.-', label = 'ADL (AUC:%2.2f)' % roc_auc_score(label, prediction))
    ax1.legend(loc = 4)
    ax1.set_xlabel('1 - Specificity')
    ax1.set_ylabel('Sensitivity')
    plt.title(title)
    fig.savefig(path)
    
    #Threshold
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    index = roc.iloc[(roc.tf-0).abs().argsort()[:1]].index
    thresh = float(thresholds[index])
    
    pred_result = np.where(prediction >= thresh, 1, 0)
    
    return pred_result

def result_original_matrix(table):
    
    tabl2_new = np.zeros((2,2))
    tabl2_new[0, 0] = int(table[1, 1])
    tabl2_new[1, 1] = int(table[0, 0])
    tabl2_new[1, 0] = int(table[1, 0]) #original False Negative
    tabl2_new[0, 1] = int(table[0, 1]) #original False Positive

    bot = ['True Positive', 'True Negative']
    left = ['Model Positive', 'Model Negative']

    df_cm = pd.DataFrame(tabl2_new, 
                         index = [i for i in left],
                         columns = [i for i in bot])
    
    result_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                                      'Accuracy','F1Score', 'AUC'])
    result_df.TP = pd.Series(int(tabl2_new[0,0]))
    result_df.FP = pd.Series(int(tabl2_new[0,1]))
    result_df.TN = pd.Series(int(tabl2_new[1,1]))
    result_df.FN = pd.Series(int(tabl2_new[1,0]))
    
    #Recall
    sen = tabl2_new[0, 0]/(tabl2_new[0, 0] + tabl2_new[1, 0])
    result_df.Sensitivity = sen
    result_df.Specificity = tabl2_new[1, 1]/(tabl2_new[0, 1] + tabl2_new[1, 1])
    
    #Precision
    ppv = tabl2_new[0, 0]/(tabl2_new[0, 0] + tabl2_new[0, 1])
    result_df.PPV = ppv
    result_df.NPV = tabl2_new[1, 1]/(tabl2_new[1, 1] + tabl2_new[1, 0])
    result_df.Accuracy = (tabl2_new[0, 0] + tabl2_new[1, 1])/table.sum()

    result_df.F1Score = 2 * (ppv * sen) / (ppv + sen)
        
    return result_df

# main
pred_result = ROC_plot(label = test_labels, prediction = prediction, title= 'DNN', path = 'DNN.JPEG')

table = confusion_matrix(test_labels, pred_result)
table_df = result_original_matrix(table)
table_df
    
path = 'DNN_result.csv'
table_df.to_csv(path, index = False)