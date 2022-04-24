# -*- coding: utf-8 -*-

#引入模組與資料
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, metrics
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

features = pd.read_csv('D:/File_X/Help/hsu/sig_var.csv')
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

def ROC_plot(label, prediction, title):
    
    #Plot
    fpr, tpr, thresholds = roc_curve(label, prediction)
    fig, ax1 = plt.subplots(1,1, figsize = (10, 10), dpi = 80)
    ax1.plot(fpr, tpr, 'b.-', label = 'ADL (AUC:%2.2f)' % roc_auc_score(label, prediction))
    ax1.legend(loc = 4)
    ax1.set_xlabel('1 - Specificity')
    ax1.set_ylabel('Sensitivity')
    plt.title(title)
    fig.savefig('Random Forest.JPEG')
    #Threshold
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    index = roc.iloc[(roc.tf-0).abs().argsort()[:1]].index
    thresh = float(thresholds[index])
    
    pred_result = np.where(predictions >= thresh, 1, 0)
    
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

pred_result = ROC_plot(label = test_labels, prediction = predictions, title= 'Random Forest')

table = confusion_matrix(test_labels, pred_result)
table_df = result_original_matrix(table)
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
