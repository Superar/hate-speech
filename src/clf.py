import pickle
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from data_preprocess import read_data
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

data_header = '../data/'

def run_cv(train,labels,debug):
    print(debug)
    
    svm = LinearSVC(random_state=0, tol=1e-5)
    scores = cross_val_score(svm, train, labels, cv=5)
    print('svm',scores)

    knn = KNeighborsClassifier(n_neighbors=50, metric='euclidean')
    scores = cross_val_score(knn, train, labels, cv=5)
    print('knn',scores)

    mlp = MLPClassifier(hidden_layer_sizes=(50,20))
    scores = cross_val_score(mlp, train, labels, cv=5)
    print('mlp',scores)

metadata = pd.read_csv('../hate-speech-dataset/annotations_metadata.csv')
_,labels = read_data(metadata,'../hate-speech-dataset/sampled_train/')

with open(data_header + 'word2vec_train.pickle', 'rb') as f:
    w2v = pickle.load(f)
run_cv(w2v,labels.values,'word2vec')

with open(data_header + 'selected_train.pickle', 'rb') as f:
    corpus = pickle.load(f)

vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(corpus)
run_cv(bow,labels.values,'bow')

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)
run_cv(tfidf,labels.values,'tfidf')