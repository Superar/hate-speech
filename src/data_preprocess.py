#! /usr/bin/env python3
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def read_data(path):
    
    filenames = os.listdir(path)
    file_ids = [os.path.splitext(f)[0] for f in filenames]

    corpus = [open(path+f).read() for f in filenames]
    vectorizer = CountVectorizer(stop_words=stopwords.words('english')) # n_features=2**12
    corpus = vectorizer.fit_transform(corpus)

    corpus_pd = pd.DataFrame(corpus.toarray(),columns=vectorizer.vocabulary_)
    
    return file_ids,corpus_pd

def select_by_corr(corpus,file_ids):
    labels = metadata[metadata['file_id'].isin(file_ids)].label
    labels = pd.get_dummies(labels)
    corr = corpus.corrwith(labels['hate'])
    # nans = corr[corr.isna()].index
    # for word in nans:
    #     print(word,sum(corpus[word]))
    # corr = corr.fillna(value=0)
    not_nans = corr[corr.notna()].index
    for word in not_nans:
        print(word,sum(corpus[word]))
    return corr
    
metadata = pd.read_csv('hate-speech-dataset/annotations_metadata.csv')
file_ids,corpus = read_data('hate-speech-dataset/sampled_train/')



select_by_corr(corpus,file_ids)