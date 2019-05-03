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
    
    # svm = LinearSVC(random_state=0, tol=1e-5)
    # scores = cross_val_score(svm, train, labels, cv=5, scoring='f1')
    # print('svm',np.mean(scores),np.std(scores))

    for k in [2,5,10,50]:
	    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
	    scores = cross_val_score(knn, train, labels, cv=5, scoring='f1')
	    print('knn',k,np.mean(scores),np.std(scores))

    # for dims in [(50,20),(50,20,5),(100,50),(100,50,10)]:
	#     mlp = MLPClassifier(hidden_layer_sizes=dims)
	#     scores = cross_val_score(mlp, train, labels, cv=5, scoring='f1')
	#     print('mlp',dims,np.mean(scores),np.std(scores))

metadata = pd.read_csv('../hate-speech-dataset/annotations_metadata.csv')
_,labels = read_data(metadata,'../hate-speech-dataset/sampled_train/')

with open(data_header + 'word2vec_train.pickle', 'rb') as f:
    w2v = pickle.load(f)
run_cv(w2v,labels.values,'word2vec')

for i in [1000]:
	print(i)

	with open(data_header + 'selected'+str(i)+'_train.pickle', 'rb') as f:
	    corpus = pickle.load(f)

	vectorizer = CountVectorizer()
	bow = vectorizer.fit_transform(corpus)
	run_cv(bow,labels.values,'bow')

	vectorizer = TfidfVectorizer()
	tfidf = vectorizer.fit_transform(corpus)
	run_cv(tfidf,labels.values,'tfidf')

