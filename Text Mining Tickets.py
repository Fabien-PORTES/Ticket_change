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


from fonction_text_mining import DeleteDictWords, MergeFeatures, Lemmatization, Tfvectorizer, vectorizerData, vectorizerDataBigram, stopWordsFR

french_dict_path = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\Louis\\Machine learning Tickets\\Etude Tickets Changes\\Data Changes\\dico fr.txt"
path_to_merge = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\prepared_df_sum.csv"
path_to_save = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\Text_mining.csv"
data = pd.read_csv(path_to_merge, sep=';', index_col = "ID")

text = list()
text = data["Summary"].tolist()

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

#data = Tfvectorizer(data)
#data_features = MergeFeatures(data, path_to_merge)

data_features.to_csv(path_to_save, sep=';', encoding='utf-8', index = False)
print("%s text mining features" %(str(data_ngram.shape)))








