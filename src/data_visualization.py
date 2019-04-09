#! /usr/bin/env python3
import os
import string
import matplotlib.pyplot as plt
import numpy as np
from nltk import word_tokenize, pos_tag
from collections import Counter
from tqdm import tqdm

DATA_PATH = 'hate-speech-dataset/all_files'

tag_counts = Counter()
for filename in tqdm(os.listdir(DATA_PATH)):
    with open(DATA_PATH + '/' + filename) as f:
        tokens = word_tokenize(f.read())
        tagged = list()
        for tok in pos_tag(tokens):
            if tok[1] not in string.punctuation and tok[1] not in ["''", '``']:
                tagged.append(tok[1])
        tag_counts.update(Counter(tagged))

labels, values = zip(*sorted(tag_counts.items(),
                             key=lambda x: x[1],
                             reverse=True))
idx = np.arange(2*len(labels), step=2)
plt.bar(idx, values, width=1)
plt.xticks(idx, labels=labels, rotation=90)
plt.xlabel('Etiquetas')
plt.ylabel('Frequência absoluta')
plt.show()
