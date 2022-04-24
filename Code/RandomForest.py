# -*- coding: utf-8 -*-

#引入模組與資料
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, metrics
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import Evaluation as eva

features = pd.read_csv('sig_var.csv')
features.head(5)

# 探索資料
print('The shape of our features is:', features.shape)
features.describe()
features.head()

# checking null
print(features.isnull().sum())

# One-Hot Encoding (將類別變數轉成數值化)
# One-hot encode the data using pandas get_dummies
# features2 = pd.get_dummies(features)
# Display the first 5 rows of the last 12 columns
# features.iloc[:,5:].head(5)

# 特徵轉換
# Labels are the values we want to predict
labels = np.array(features['ADL-group'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('ADL-group', axis = 1)
features= features.drop('ID', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# 將資料分成訓練組及測試組
# Using Skicit-learn to split data into training and testing sets
# from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

sum(test_labels)/len(test_features)
sum(train_labels)/len(train_features)

# =============================================================================
# np.where(np.isnan(train_features))
# np.where(np.isnan(train_labels))
# train_features = np.nan_to_num(train_features)
# 
# print(train_features.dtype)
# print(train_labels.dtype)
# train_features = np.float32(train_features)
# train_labels = np.float32(train_labels)
# =============================================================================

# 使用隨機森林演算法(1000顆樹)
# from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees

#1000 tree, 50 node size
rf = RandomForestRegressor(n_estimators = 1000,
                           min_samples_leaf = 100)  #can modify the number of trees
# Train the model on training data
rf.fit(train_features, train_labels)

# 進行預測
# Use the forest's predict method on the test data
# test_features = np.nan_to_num(test_features)
predictions = rf.predict(test_features)

# Evaluation main
pred_result = eva.ROC_plot(label = test_labels, prediction = predictions, title= 'Random Forest')

table = confusion_matrix(test_labels, pred_result)
table_df = eva.result_original_matrix(table)
table_df
    
path = 'Random_Forest_result.csv'
table_df.to_csv(path, index = False)

###################################################################    
# 繪製決策樹(1000顆樹)
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('D:/File_X/Help/hsu/tree.png')

# 呈現決策樹圖片
from IPython.display import Image
from IPython.core.display import HTML 
PATH = "tree.png"
Image(filename = PATH , width=1000, height=1000)

# 變數特徵的重要程度
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# 視覺化變數特徵的重要程度
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
