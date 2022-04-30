# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#data
features = pd.read_csv('D:/File_X/Help/hsu/sig_var.csv')
# features.head(5)

# 探索資料
# print('The shape of our features is:', features.shape)
# features.describe()
# features.head()

# checking null
# print(features.isnull().sum())

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
# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)

# sum(test_labels)/len(test_features)
# sum(train_labels)/len(train_features)

x_train_path = 'train_features.csv'
pd.DataFrame(train_features).to_csv(x_train_path, index = False)

x_test_path = 'test_features.csv'
pd.DataFrame(test_features).to_csv(x_test_path, index = False)

y_train_path = 'train_labels.csv'
pd.DataFrame(train_labels).to_csv(y_train_path, index = False)

y_test_path = 'test_labels.csv'
pd.DataFrame(test_labels).to_csv(y_test_path, index = False)

def read_features(x_train = x_train_path, y_train = y_train_path, x_test = x_test_path, y_test = y_test_path):

    train_features = pd.read_csv(x_train)
    train_features = train_features.to_numpy()
    
    test_features = pd.read_csv(x_test)
    test_features = test_features.to_numpy()
    
    train_labels = pd.read_csv(y_train)
    train_labels = train_labels.to_numpy()
    
    test_labels = pd.read_csv(y_test)
    test_labels = test_labels.to_numpy()
    
    return train_features, train_labels, test_features, test_labels


    
    

    
