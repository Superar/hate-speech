import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from data_preprocess import read_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('processed_train.csv')
metadata = pd.read_csv('hate-speech-dataset/annotations_metadata.csv')
_,labels = read_data(metadata,'hate-speech-dataset/sampled_train/')

svm = LinearSVC(random_state=0, tol=1e-5)
scores = cross_val_score(svm, train.values, labels.values, cv=5)
print('svm',scores)

knn = KNeighborsClassifier(n_neighbors=100, metric='euclidean')
scores = cross_val_score(knn, train.values, labels.values, cv=5)
print('knn',scores)
