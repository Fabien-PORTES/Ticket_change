#-*- coding: utf-8 -*-
import json, sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from sklearn import cross_validation
from sklearn import svm


to_parse = " ".join(sys.argv[1:])
user_input = json.loads(to_parse)
path = user_input['path_data']
path_to_save = user_input['path_to_save']
important_feature_file = user_input['important_feature_file']


#path = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\cleaned_data_changes.csv"
#path_to_save = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\Ticket_change_predicted.csv"
#important_feature_file = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\Important_features.txt"

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

data = pd.read_csv(path, sep=';', index_col = "ID", low_memory = False)

target_list = ['delay_48h_bin', "delay_14h_bin"]
to_encode = ['anciennete_Modified',
       'INST_Task_Assignee_Modified', 'Support_Group_Name+_Modified',
       'INST_techno_gpe_Modified', 'TOC_code_Modified',
       'TOC_level_Modified', 'INST_Task_Assignee_Modified', 'Performance_Rating_Modified',
       'INST_desc_techno_Modified', 'EIST_Domain_ICT_Modified']
#print(data[to_encode])
print(data.shape)
data[to_encode] = data[to_encode].apply(LabelEncoder().fit_transform)

target_14h = False
if target_14h is True:
    target = 'delay_14h_bin'
else:
    target = 'delay_48h_bin'

# Echantillon apprentissage / test
non_predictive_var = target_list + ['Request_For_Change_Time', "Summary"]

X = data[[str(c) for c in data.columns if c not in non_predictive_var]]  # variables explicatives
y = data[target] # variable a modeliser


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)


best_param = {'solver': 'liblinear', 'penalty': 'l2', 'C': 0.05}

param=[{"C":[0.1, 0.05, 0.01], "penalty" : ["l2"], "solver" : ["liblinear"]}]
["newton-cg", "lbfgs", "liblinear", "sag"]
estim = LogisticRegression(class_weight="balanced")
logit = GridSearchCV(estim, param,cv=2,n_jobs=1, scoring = "roc_auc") 
data_logit=logit.fit(X_train,y_train)
best_param = data_logit.best_params_
print(data_logit.best_params_)

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
modelEvaluation(y_train.values, Xpred)

#scores = cross_validation.cross_val_score(estim, X[important_features], y, cv=10, scoring = 'roc_auc')
#print(sorted(scores))
#print("ROC_auc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

data['target_predicted'] = estim.predict(data[important_features])
data["Failure_probability"] = estim.predict_proba(data[important_features])[:,1]

to_save = ['target_predicted', 'Failure_probability'] + important_features
data[to_save].to_csv(path_to_save, sep = ";", encoding = 'utf-8')


from sklearn import svm
from sklearn.preprocessing import scale

# SVM -----------
param= {"class_weight":['balanced', None],
        "kernel" : ['linear', 'rbf'],
        "shrinking" : [True, False],
        "C" : [ 0.05, 0.005, 0.01]}

#estim = svm.SVC()
#logit = GridSearchCV(estim, param, cv=2, n_jobs=1, scoring = "roc_auc") 
#data_logit=logit.fit(X_train,y_train)
#best_param = data_logit.best_params_
#print(best_param)
#best_param = {'kernel': 'linear', 'class_weight': 'balanced', 'C': 0.01, 'shrinking': True}
best_param = {'class_weight': 'balanced', 'shrinking': False, 'C': 0.01, 'kernel': 'linear', "probability" : True}
estim = svm.SVC(**best_param)
sv = estim.fit(X_train[important_features], y_train)
ypred = sv.predict(X_test[important_features])
x_train_pred = sv.predict(X_train[important_features])
modelEvaluation(y_test.values, ypred)
modelEvaluation(y_train.values, x_train_pred)


#scores = cross_validation.cross_val_score(estim, X, y, cv=10, scoring = 'roc_auc')
#print(sorted(scores))
#print("ROC_auc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#param= {"class_weight":['balanced'],
#        "loss" : ['hinge'],
#        "penalty" : ["l2"],
#        "dual" : [True],
#        "C" : [30, 70, 50] }
#        
#estim = LinearSVC()
#logit = GridSearchCV(estim, param, cv=2, n_jobs=1, scoring = "roc_auc") 
#data_logit=logit.fit(X_train,y_train)
#best_param = data_logit.best_params_
#print(best_param)
#estim = LinearSVC(**best_param)
#yolo = scale(X_train[important_features])
#sv = estim.fit(scale(X_train[important_features]), y_train)
#ypred = sv.predict(scale(X_test[important_features]))
#x_train_pred = sv.predict(scale(X_train[important_features]))
#modelEvaluation(y_test.values, ypred)
#modelEvaluation(y_train.values, x_train_pred)

## PERCEPTRON -----------
#from sklearn.linear_model import Perceptron
#
#param= {"class_weight":['balanced', None],
#        "penalty" : [None, 'l2',  'l1', 'elasticnet'],
#        "shuffle" : [True, False],
#        "alpha" : [0.000001, 0.00001, 0.0001, 0.001, 0.01] }
#
#estim = Perceptron()
#
#logit = GridSearchCV(estim, param,cv=2,n_jobs=1, scoring = "roc_auc") 
#data_logit=logit.fit(X_train,y_train)
#best_param = data_logit.best_params_
#print(best_param)
#
#estim = Perceptron(**best_param)
#percep = estim.fit(X_train[important_features], y_train)
#ypred = percep.predict(X_test[important_features])
#modelEvaluation(y_test.values, ypred)
#

