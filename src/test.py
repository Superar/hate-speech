import pickle
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from data_preprocess import read_data
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

data_header = '../data/'

def run_test(train,labels_train,test,labels_test,clf):

	clf = clf.fit(train,labels_train)
	predictions = clf.predict(test)

	f1 = f1_score(labels_test, predictions, average='binary')  
	p = precision_score(labels_test,predictions)
	r = recall_score(labels_test,predictions)
	a = accuracy_score(labels_test,predictions)

	print("%.4f & %.4f & %.4f & %.4f" % (f1,p,r,a))
	
metadata = pd.read_csv('../hate-speech-dataset/annotations_metadata.csv')
_,labels_train = read_data(metadata,'../hate-speech-dataset/sampled_train/')
_,labels_test = read_data(metadata,'../hate-speech-dataset/sampled_test/')

with open(data_header + 'word2vec_train.pickle', 'rb') as f:
    train = pickle.load(f)
with open(data_header + 'word2vec_test.pickle', 'rb') as f:
    test = pickle.load(f)
print('w2v + mlp')
mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 10))
run_test(train,labels_train,test,labels_test,mlp)

i = 500

with open(data_header + 'selected'+str(i)+'_train.pickle', 'rb') as f:
	corpus_train = pickle.load(f)

with open(data_header + 'selected'+str(i)+'_test.pickle', 'rb') as f:
	corpus_test = pickle.load(f)

print('bow + mlp')
vectorizer = CountVectorizer()
vectorizer = vectorizer.fit(corpus_train)
train = vectorizer.transform(corpus_train)
test = vectorizer.transform(corpus_test)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50))
run_test(train,labels_train,test,labels_test,mlp)

print('tf-idf + svm')
vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(corpus_train)
train = vectorizer.transform(corpus_train)
test = vectorizer.transform(corpus_test)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 10))
run_test(train,labels_train,test,labels_test,mlp)
