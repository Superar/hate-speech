#! /usr/bin/env python3
import os
import gensim
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

data_header = '../data/'

def read_data(metadata,path):
    filenames = os.listdir(path)
    
    file_ids = [os.path.splitext(f)[0] for f in filenames]
    labels = metadata[metadata['file_id'].isin(file_ids)].label
    labels = pd.get_dummies(labels)['hate']
    
    corpus = [open(path+f).read() for f in filenames]
    
    return corpus,labels

def word2vec(data,k=300,path='../data/model.bin'):
    corpus = [word_tokenize(sentence) for sentence in data]
    try:
        model = Word2Vec.load(path)
    except:
        model = Word2Vec(corpus, size=k, window=5, min_count=1, workers=4)
        model.save(path)

    transformed_corpus = []
    for example in corpus:
        vec = np.zeros(k)
        for word in example:
            vec += model[word]
        vec /= k
        transformed_corpus.append(vec)

    return transformed_corpus

def select_by_corr(corpus,labels,top_n=1000):
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    bow = vectorizer.fit_transform(corpus)

    corrs = []
    for col in bow.toarray().T:
        corr, _ = pearsonr(col, labels)
        corrs.append(corr)

    vocab = np.asarray(vectorizer.get_feature_names())
    sorted_idxs = np.argsort(np.abs(corrs))[-top_n:]
    top_sorted = vocab[sorted_idxs]
    
    selected_corpus = []
    for sentence in corpus:
        s = ''
        sentence = word_tokenize(sentence)
        for word in sentence:
            if word in top_sorted:
                s += word + ' '
        selected_corpus.append(s)

    return selected_corpus
    
def main():
    metadata = pd.read_csv('../hate-speech-dataset/annotations_metadata.csv')
    corpus,labels = read_data(metadata,'../hate-speech-dataset/sampled_train/')


    corpus1 = word2vec(corpus)
    with open(data_header + 'word2vec_train.pickle', 'wb') as f:
        pickle.dump(corpus1, f, pickle.HIGHEST_PROTOCOL)


    for i in [500,1000,2000,3000]:
        print(i)
        corpus2 = select_by_corr(corpus,labels,i)
        with open(data_header + 'selected'+str(i)+'_train.pickle', 'wb') as f:
            pickle.dump(corpus2, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()