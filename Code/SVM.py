# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:35:41 2022

@author: Lin
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
%matplotlib inline

model = SVC(kernel='linear')
model.fit(train_features,train_labels)

# 進行預測
# Use the forest's predict method on the test data
# test_features = np.nan_to_num(test_features)
predictions = model.predict(test_features)

#載入classification report & confusion matrix
print(confusion_matrix(test_labels, predictions))
print('\n')
print(classification_report(test_labels, predictions))

#載入GridSearchCV
#GridSearchCV是建立一個dictionary來組合要測試的參數
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
#GridSearchCV算是一個meta-estimator，參數中帶有estimator，像是SVC。
# 重點是會創造一個新的estimator，但又表現得一模一樣，也就是estimator=SVC時，就是作為分類器
#Verbose可設定為任一整數，它只是代表數字越高，文字解釋越多
grid = GridSearchCV(SVC(),param_grid,verbose=3)

#利用剛剛設定的參數來找到最適合的模型
grid.fit(train_features, train_labels)

#顯示最佳estimator參數
grid.best_estimator_

#利用剛剛的最佳參考再重新預測測試組
grid_predictions = grid.predict(X_test)

#評估新參考的預測結果好壞
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))

#Evaluation

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

pred_result = ROC_plot(label = test_labels, prediction = predictions, title= 'SVM')

table = confusion_matrix(test_labels, pred_result)
table_df = result_original_matrix(table)
table_df
    
path = 'SVM_result.csv'
table_df.to_csv(path, index = False)