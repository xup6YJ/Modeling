# -*- coding: utf-8 -*-

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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance

import Model as md

features = pd.read_csv('IPTW.csv')


# ID,age,sex,BMI,edu,marriage,Caregiver,smoking,drinking,CCI,inADL,GDS5,walk,stratify,cam,BradenScale,EQ5D,tube,urinary,days,readmission,PRE_1,weight
labels = np.array(features['readmission'])   #convert to your y column name

features= features.drop('readmission', axis = 1)  #convert to your y column name
features= features.drop('ID', axis = 1)
features= features.drop('weight', axis = 1)
features= features.drop('PRE_1', axis = 1)
feature_list = list(features.columns)



def dnn_model():
    
    model = Sequential()
    model.add(Dense(units=512, activation='relu', input_dim=len(feature_list)))  #convert to your own input_dim
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



rnn_train_features = np.array(features)
rnn_train_features = np.reshape(rnn_train_features, (rnn_train_features.shape[0], rnn_train_features.shape[1], 1))
rnn_train_shape = (rnn_train_features.shape[1], rnn_train_features.shape[2])

def rnn_model():
    
    model = Sequential()
    model.add(SimpleRNN(units = 256, input_shape = rnn_train_shape, activation = 'relu'))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units = 16, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    # model.summary()
    
    
    model.compile(loss='binary_crossentropy', 
                  optimizer  = optimizers.Adam (learning_rate = 0.001, name = 'adam'), metrics = ['acc'])
    
    return model



def model_importance(model_type):
    
    if model_type == 'DNN':

        model = dnn_model()   
            
        clf = KerasRegressor(build_fn=dnn_model, epochs=10, batch_size=16)
        clf.fit(features, labels)

        perm = PermutationImportance(clf, random_state=1).fit(features,labels)
        w = eli5.show_weights(perm, feature_names = features.columns.tolist())

        result = pd.read_html(w.data)[0]
        print(result)
        
        
    elif model_type == 'RNN':

        model = rnn_model()
        clf = KerasRegressor(build_fn=dnn_model, epochs=10, batch_size=16)
        clf.fit(features, labels)
        
        perm = PermutationImportance(clf, random_state=1).fit(features, labels)
        w = eli5.show_weights(perm, feature_names = features.columns.tolist())
        
        result = pd.read_html(w.data)[0]
        print(result)
        
    #ML
    elif model_type == 'Random Forest':
        # Instantiate model with 1000 decision trees, 50 node size
        model = md.rf_model()
        history = model.fit(features, labels.ravel())
        
        imps = permutation_importance(model, features, labels)
        imps_list = imps.importances_mean
    
        print('Result of ', model_type)
        print('Feature importance', list(zip(feature_list, imps_list)))
    
    elif model_type == 'SVM':
        model = md.svm_model()
        history = model.fit(features,labels.ravel())
        
        imps = permutation_importance(model, features, labels)
        imps_list = imps.importances_mean
    
        print('Result of ', model_type)
        print('Feature importance', list(zip(feature_list, imps_list)))
        
    elif model_type == 'Logistic':
        model = md.logistic_reg()
        model.fit(features,labels.ravel())
        
        imps = permutation_importance(model, features, labels)
        imps_list = imps.importances_mean
    
        print('Result of ', model_type)
        print('Feature importance', list(zip(feature_list, imps_list)))
        
    elif model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=3)
        history = model.fit(features,labels.ravel())
        
        imps = permutation_importance(model, features, labels)
        imps_list = imps.importances_mean
    
        print('Result of ', model_type)
        print('Feature importance', list(zip(feature_list, imps_list)))
        
    elif model_type == 'NBC':
        model = GaussianNB()
        history = model.fit(features,labels.ravel())
        
        imps = permutation_importance(model, features, labels)
        imps_list = imps.importances_mean
    
        print('------- Result of ', model_type, ' -------')
        print('Feature importance', list(zip(feature_list, imps_list)))




model_importance(model_type = 'DNN')

model_importance(model_type = 'RNN')

model_importance(model_type = 'Random Forest')

model_importance(model_type = 'SVM')

model_importance(model_type = 'Logistic')

model_importance(model_type = 'KNN')

model_importance(model_type = 'NBC')
