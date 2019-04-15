#! /usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def read_data(metadata,path):
    filenames = os.listdir(path)
    
    file_ids = [os.path.splitext(f)[0] for f in filenames]
    labels = metadata[metadata['file_id'].isin(file_ids)].label
    labels_pd = pd.get_dummies(labels)['hate']
    
    corpus = [open(path+f).read() for f in filenames]
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    corpus = vectorizer.fit_transform(corpus)
    corpus_pd = pd.DataFrame(corpus.toarray(),columns=vectorizer.get_feature_names())
    
    return corpus_pd,labels_pd

def select_by_corr(corpus,labels,top_n=1000):
    corrs = []
    for col in corpus:
        col = corpus[col].values
        corr, _ = pearsonr(col, labels)
        corrs.append(corr)

    cols = corpus.columns
    sorted_idxs = np.argsort(np.abs(corrs))[-top_n:]
    top_sorted = cols[sorted_idxs]

    return top_sorted
    
metadata = pd.read_csv('hate-speech-dataset/annotations_metadata.csv')
corpus,labels = read_data(metadata,'hate-speech-dataset/sampled_train/')

selected_feats = select_by_corr(corpus,labels)
corpus = corpus[selected_feats]
corpus.to_csv('processed_train.csv')