# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

x_train_path = 'train_features.csv'
x_test_path = 'test_features.csv'
y_train_path = 'train_labels.csv'
y_test_path = 'test_labels.csv'
exv_x_path = 'ex_features.csv'
exv_y_path = 'ex_labels.csv'


class dat:
    def __init__(self, path):
        
        self.path = path
        self.features = pd.read_csv(self.path)  #convert path
        self.x, self.y = self.clean_data()
              
    def clean_data(self):
        # featuring
        labels = np.array(self.features['ADL-group'])   #convert to your y column name
        # Remove the labels from the features
        # axis 1 refers to the columns
        data= self.features.drop('ADL-group', axis = 1)  #convert to your y column name
        data= data.drop('ID', axis = 1)
        # Saving feature names for later use
        feature_list = list(data.columns)
        # Convert to numpy array
        data = np.array(data)
        
        return data, labels
    
    def shuffle_dat(self):
        train_features, test_features, train_labels, test_labels = train_test_split(self.x, self.y, test_size = 0.3)
        
        return train_features, test_features, train_labels, test_labels
    

def read_features(x_train = x_train_path, y_train = y_train_path, x_test = x_test_path, y_test = y_test_path):

    train_features = pd.read_csv(x_train)
    train_features = np.asarray(train_features).astype(np.float32)
    
    test_features = pd.read_csv(x_test)
    test_features = np.asarray(test_features).astype(np.float32)
    
    train_labels = pd.read_csv(y_train)
    train_labels = np.asarray(train_labels).astype(np.float32)
    
    test_labels = pd.read_csv(y_test)
    test_labels = np.asarray(test_labels).astype(np.float32)
    
    return train_features, train_labels, test_features, test_labels

def read_external_data(x = exv_x_path, y = exv_y_path):
    
    exv_features = pd.read_csv(x)
    exv_features = np.asarray(exv_features).astype(np.float32)
    
    exv_labels = pd.read_csv(y)
    exv_labels = np.asarray(exv_labels).astype(np.float32)
    
    return exv_features, exv_labels


if __name__ == "__main__":
    data = dat('internal.csv')
    train_features, test_features, train_labels, test_labels = data.shuffle_dat()
    
    data2 = dat('external.csv')
    features, labels = data2.clean_data()
    
    pd.DataFrame(train_features).to_csv(x_train_path, index = False)
    pd.DataFrame(test_features).to_csv(x_test_path, index = False)
    pd.DataFrame(train_labels).to_csv(y_train_path, index = False)
    pd.DataFrame(test_labels).to_csv(y_test_path, index = False)
    
    pd.DataFrame(features).to_csv(exv_x_path, index = False)
    pd.DataFrame(labels).to_csv(exv_y_path, index = False)
   
