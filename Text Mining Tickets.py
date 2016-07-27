# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:35:05 2016

@author: lmellior
"""
import os
import nltk.corpus
from nltk.corpus import stopwords
from nltk.corpus import words
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import bigrams
from sklearn.feature_extraction.text import CountVectorizer


###############################################################################
#                                                                             #
#                        CountVectorizer  CHANGES                             #
#                                                                             #
###############################################################################


from fonction_text_mining import MergeFeatures, DeleteDictWords, Lemmatization, Tfvectorizer, vectorizerData, vectorizerDataBigram, stopWordsFR

french_dict_path = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\Louis\\Machine learning Tickets\\Etude Tickets Changes\\Data Changes\\dico fr.txt"
path_to_merge = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\prepared_df_sum.csv"
path_to_save = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\Text_mining.csv"
important_feature_file = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\Important_feature.txt"

target_list = ["delay_48h_bin", "delay_14h_bin"]
non_predictive_var = target_list + ['Request_For_Change_Time', "Summary"]

target_14h = False
if target_14h is True:
    target = "delay_14h_bin"
else:
    target = "delay_48h_bin"

data_init = pd.read_csv(path_to_merge, sep=';', index_col = "ID")
print(data_init.describe())

text = list()
text = data_init["Summary"].tolist()

data = text

# Lemmatization
data = [Lemmatization(i) for i in data]  

# Vectorizer 1-gram : vectorizerData(data, min_freq):
data_unigram = vectorizerData(data,5)
#print(data_unigram)
# Vectorizer bi-gram : vectorizerDataBigram(data, min_freq):
data_bigram = vectorizerDataBigram(data,5)

# Stop words francais
data_unigram = stopWordsFR(data_unigram)

# dico : DeleteDictWords(data,dictEN, dictFR)
#data_unigram = DeleteDictWords(data_unigram, 1, 1, french_dict_path)

# Merge des n-gram
data_ngram = pd.concat([data_unigram,data_bigram], axis=1)

# Merge avec le modèle simple
data_features = MergeFeatures(data_ngram, path_to_merge)
# Save CSV
#os.chdir( "D:/Users/lmellior/Desktop/Sogeti/KORMIN/Etude Tickets Changes/code Python/TM csv")
data_features.to_csv(path_to_save , sep=';', encoding='utf-8', index = False)

# Pondération avec TF IDF tdf 
get_imp_text_features = True
if get_imp_text_features is True:
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFE
    from sklearn.grid_search import GridSearchCV
    #param=[{"C":[0.01, 0.001, 0.005], "solver" : ["newton-cg"]}]
    #best_param = {"C" : 0.001, "penalty" : "l2", "solver" : "newton-cg"}
    best_param = {'penalty': 'l2', 'C': 0.05, 'solver': 'liblinear'}
    #estim = LogisticRegression(class_weight="balanced")
    #logit = GridSearchCV(estim, param_grid = param, cv=3, n_jobs=1, scoring = "roc_auc")
    #data_logit=logit.fit(data_ngram, data_init["delay_48h_bin"])
    #best_param = data_logit.best_params_
    
    
    estim = LogisticRegression(class_weight="balanced", **best_param)
    print(estim)
    data_logit=estim.fit(data_ngram, data_init[target])
    
    selecteur = RFE(estimator=estim, n_features_to_select = 300)
    sol = selecteur.fit(data_ngram, data_init[target])
    important_features = data_ngram.columns[sol.support_].tolist()
    important_features = [c for c in data_init.columns.tolist() if c not in non_predictive_var] + important_features
    with open(important_feature_file, "w") as text_file:
        text_file.write("\n".join(str(a) for a in important_features))

#data = Tfvectorizer(data)
#data_features = MergeFeatures(data, path_to_merge)

data_features.to_csv(path_to_save, sep=';', encoding='utf-8', index = False)
print("%s text mining features" %(str(data_ngram.shape)))








