# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:15:37 2022

@author: Admin
"""

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

def clean_data():
    # featuring
    labels = np.array(features['readmission'])   #convert to your y column name
    # Remove the labels from the features
    # axis 1 refers to the columns
    data= features.drop('readmission', axis = 1)  #convert to your y column name
    data= data.drop('ID', axis = 1)
    # Saving feature names for later use
    feature_list = list(data.columns)
    # Convert to numpy array
    data = np.array(data)
    
    return data, labels
    
    
features = pd.read_csv('IPTW.csv')
x, y = clean_data()

#Total ration of training/ Internal validation/ External validation = 70% /15% /15%
#External
x_train, eva_x_test, y_train, eva_y_test = train_test_split(x, y, test_size = 0.15, stratify=y)

#Internal
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.18, stratify=y_train)  

pd.DataFrame(x_train).to_csv(x_train_path, index = False)
pd.DataFrame(x_test).to_csv(x_test_path, index = False)
pd.DataFrame(y_train).to_csv(y_train_path, index = False)
pd.DataFrame(y_test).to_csv(y_test_path, index = False)

pd.DataFrame(eva_x_test).to_csv(exv_x_path, index = False)
pd.DataFrame(eva_y_test).to_csv(exv_y_path, index = False)



   
