# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:32:44 2016

@author: lmellior
"""
#Albertus gaelle

import os
import re
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
from sklearn.feature_extraction.text import TfidfVectorizer

###############################################################################
#                                                                             #
#                          FONCTIONS TEXTE MINING                             #
#                                                                             #
###############################################################################

def Lemmatization(text):
    lemma = [lemmatizer.lemmatize(word) for word in text.split(' ')]   
    lemma_data = ' '.join(lemma)
    return lemma_data

def vectorizerData(data, min_freq):
    vectorizer = CountVectorizer(analyzer='word', min_df = min_freq, lowercase=True,
                stop_words='english',token_pattern='(?u)\\b\\w\\w+\\b', binary=True,
                ngram_range=(1,1))
                
    X = vectorizer.fit_transform(data)  
    return pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names() )
    
def vectorizerDataBigram(data, min_freq):
    vectorizer = CountVectorizer(analyzer='word', min_df = min_freq, lowercase=True,
                stop_words='english',token_pattern='(?u)\\b\\w\\w+\\b', binary=True,
                ngram_range=(2,2))
                
    X = vectorizer.fit_transform(data)
    df = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())

    regex = re.compile(r'\d')
    new_list = [s for s in vectorizer.get_feature_names() if not regex.match(s)]

    return pd.DataFrame(df, columns = new_list)


def stopWordsFR(data):
    stop_fr = stopwords.words('french')
    words = [i for i in data.columns.values if i not in stop_fr] 
    return pd.DataFrame(data, columns = words)
    

def DeleteDictWords(data,dictEN, dictFR, french_dict_path):
    
    if dictEN == 1:
        english_dict = words.words('en')
    else:
        english_dict=[]
        
    if dictFR == 1:
        french_dict = [line.rstrip('\n') for line in open(french_dict_path, 'r'
                                                            ,encoding='utf-8')]   
    else : 
        french_dict = []
    
    dico = english_dict + french_dict
    col_dico = [i for i in data.columns.values if i in dico]
      
    return pd.DataFrame(data, columns = col_dico)            

def MergeFeatures(data, path_non_text_feature):
    dataChanges = pd.read_csv(path_non_text_feature, sep=';') 
    #dataChanges = pd.read_csv('DataChanges_8913_TicketsCha_wd.csv',engine='python', sep=';') # Ajout weekday
    #dataChanges = dataChanges.drop('Unnamed: 0', axis = 1)
    return pd.concat([dataChanges,data], axis=1)


def Tfvectorizer(data):
    Tfvectorizer = TfidfVectorizer(min_df = 3)  
    Y = Tfvectorizer.fit_transform(data)
    return pd.DataFrame(Y.toarray(), columns = Tfvectorizer.get_feature_names() )






