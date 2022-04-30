
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Evaludation Function
def ROC_plot(label, prediction, title, path, plot = False, save_pic = False):
    
    #Plot
    fpr, tpr, thresholds = roc_curve(label, prediction)
    
    if plot == True:
        fig, ax1 = plt.subplots(1,1, figsize = (10, 10), dpi = 80)
        ax1.plot(fpr, tpr, 'b.-', label = 'ADL (AUC:%2.2f)' % roc_auc_score(label, prediction))
        ax1.legend(loc = 4)
        ax1.set_xlabel('1 - Specificity')
        ax1.set_ylabel('Sensitivity')
        plt.title(title)
    
    if save_pic == True:
        fig.savefig(path)
    
    #Threshold
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    index = roc.iloc[(roc.tf-0).abs().argsort()[:1]].index
    thresh = float(thresholds[index])
    
    pred_result = np.where(prediction >= thresh, 1, 0)
    auc = roc_auc_score(label, prediction)
    
    return pred_result, auc


def result_original_matrix(table, auc_score):
    
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
    result_df.AUC = auc_score
    result_df.F1Score = 2 * (ppv * sen) / (ppv + sen)
        
    return result_df