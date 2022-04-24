# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import Evaluation as eva
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

#Evaluation main
pred_result = eva.ROC_plot(label = test_labels, prediction = predictions, title= 'SVM')

table = confusion_matrix(test_labels, pred_result)
table_df = eva.result_original_matrix(table)
table_df
    
path = 'SVM_result.csv'
table_df.to_csv(path, index = False)
