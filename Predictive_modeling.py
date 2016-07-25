#-*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from sklearn.feature_selection import RFE

from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

from sklearn import svm
# In[2]:

def modelEvaluation(y_test,y_pred):

    confusion = metrics.confusion_matrix(y_test,y_pred)
    TP = confusion[1,1]
    TN = confusion[0,0]
    FP = confusion[0,1]
    FN = confusion[1,0]
    
    accuracy = (TP + TN) / float(TP + TN + FP + FN)
    sensitivity = TP / float(TP + FN) #true positive rate, recall
    specificity = TN / float(TN + FP)
    false_positive_rate = FP / float(TN + FP)
    precision = TP / float(TP + FP)   
    
    confusion = pd.crosstab(y_test,y_pred,rownames =['Observations'], 
                                        colnames =['Prediction'])
   
    print('#==========================================#''\n')
    print('          -- METRICS EVALUATION --          ''\n')
    print('    accuracy : ', accuracy)
    print('    sensitivity : ', sensitivity)
    print('    specificity : ', specificity)
    print('    false positive rate : ',false_positive_rate )
    print('    precision : ', precision)
    print('\n''#==========================================#''\n')
    print('           -- Confusion matrix --          ''\n')
    print(confusion)
    print('\n''#==========================================#''\n')
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    print(thresholds)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[3]:

path = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\Text_mining.csv"
path_to_save = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\Ticket_change_predicted.csv"
important_feature_file = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\Important_feature.txt"

data = pd.read_csv(path, sep=';', index_col = "ID", low_memory = False)
to_encode = ['target_actual_effective_bin', 'anciennete_Modified',
       'INST_Task_Assignee_Modified', 'Support_Group_Name+_Modified',
       'INST_techno_gpe_Modified', 'TOC_code_Modified',
       'TOC_level_Modified', 'INST_Task_Assignee_Modified', 'Performance_Rating_Modified',
       'INST_desc_techno_Modified']
#print(data[to_encode])
print(data.shape)
data[to_encode] = data[to_encode].apply(LabelEncoder().fit_transform)


# Echantillon apprentissage / test
non_predictive_var = ['target_actual_effective_bin', 'Request_For_Change_Time', "Summary"]
target = 'target_actual_effective_bin'
X = data[[str(c) for c in data.columns if c not in non_predictive_var]]  # variables explicatives
y = data[target] # variable a modeliser


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)


best_param = {'solver': 'liblinear', 'penalty': 'l2', 'C': 0.45}
estim = LogisticRegression(class_weight="balanced", **best_param)

#backward logistic regresssion with best params to get most important features
#selecteur = RFE(estimator=estim, n_features_to_select = 200)
#sol = selecteur.fit(X,y)
#important_features = X.columns[sol.support_]

with open(important_feature_file) as f:
    important_features = [x.strip('\n') for x in f.readlines()]
#with open(important_feature_file, "w") as text_file:
#    text_file.write("\n".join(str(a) for a in important_features))

print("%s important features" %str(len(important_features)))
data_logit=estim.fit(X_train[important_features], y_train)

Xpred = estim.predict(X_train[important_features])
log_pred = data_logit.predict(X_test[important_features])
modelEvaluation(y_test.values, log_pred)
#modelEvaluation(y_train.values, Xpred)

scores = cross_validation.cross_val_score(estim, X[important_features], y, cv=10, scoring = 'roc_auc')
print(scores)
print("ROC_auc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

data['target_predicted'] = estim.predict(data[important_features])
data["Failure_probability"] = estim.predict_proba(data[important_features])[:,1]

data.to_csv(path_to_save, sep = ";", encoding = 'utf-8')



