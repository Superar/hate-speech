#! /usr/bin/env python3
import os
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from collections import Counter
from tqdm import tqdm


def visualize_class_balance(data_path):
    metadata = pd.read_csv(data_path + '/annotations_metadata.csv')
    class_counts = metadata['label'].value_counts(normalize=True)
    percentage_strings = class_counts.round(4) * 100
    percentage_strings = percentage_strings.astype('str') + '%'

    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=percentage_strings)
    plt.legend(class_counts.index)
    plt.show()


def visualize_tags(data_path):
    tag_counts = Counter()
    for filename in tqdm(os.listdir(data_path + '/all_files')):
        with open(data_path + '/all_files/' + filename) as f:
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
    plt.figure()
    plt.bar(idx, values, width=1, edgecolor='black')
    plt.xticks(idx, labels=labels, rotation=90)
    plt.xlabel('Etiquetas')
    plt.ylabel('Frequência absoluta')
    plt.show()


def visualize_polarities(data_path, lexicon_path):
    positive_path = lexicon_path + '/positive.txt'
    negative_path = lexicon_path + '/negative.txt'

    positive_f = open(positive_path, 'r')
    negative_f = open(negative_path, 'r')
    positive_words = {w.strip() for w in positive_f.readlines()}
    negative_words = {w.strip() for w in negative_f.readlines()}
    positive_f.close()
    negative_f.close()

    metadata = pd.read_csv(data_path + '/annotations_metadata.csv')

    polarity_counts = pd.DataFrame(0, index=['positive', 'negative', 'total'],
                                   columns=metadata['label'].unique())

    for _, file_ in tqdm(metadata.iterrows()):
        filepath = data_path + '/all_files/' + file_['file_id'] + '.txt'
        class_ = file_['label']

        with open(filepath) as f:
            tokens = word_tokenize(f.read())
            positive_count = len(set(tokens).intersection(positive_words))
            negative_count = len(set(tokens).intersection(negative_words))

            polarity_counts.loc['positive', class_] += positive_count
            polarity_counts.loc['negative', class_] += negative_count
            polarity_counts.loc['total', class_] += len(tokens)

    positive_rate = polarity_counts.loc['positive', :] / polarity_counts.loc['total', :]
    positive_rate *= 100
    negative_rate = polarity_counts.loc['negative', :] / polarity_counts.loc['total', :]
    negative_rate *= 100
    idx = np.arange(2*len(polarity_counts.columns), step=2)
    plt.figure()
    plt.bar(idx, positive_rate,
            width=0.5, color='green', edgecolor='black')
    plt.bar(idx+0.5, negative_rate,
            width=0.5, color='red', edgecolor='black')
    plt.xticks(ticks=idx+0.25, labels=polarity_counts.columns)
    plt.xlabel('Classes')
    plt.ylabel('Rate of polarity words (%)')
    plt.legend(['positive', 'negative'])
    plt.show()


def visualize_negation(data_path, lexicon_path):
    negation_path = lexicon_path + '/negation.txt'

    negation_f = open(negation_path, 'r')
    negation_words = {w.strip() for w in negation_f.readlines()}
    negation_f.close()

    metadata = pd.read_csv(data_path + '/annotations_metadata.csv')

    negation_counts = pd.DataFrame(0, index=['negation', 'total'],
                                   columns=metadata['label'].unique())

    for _, file_ in tqdm(metadata.iterrows()):
        filepath = data_path + '/all_files/' + file_['file_id'] + '.txt'
        class_ = file_['label']

        with open(filepath) as f:
            tokens = word_tokenize(f.read())
            negation_count = len(set(tokens).intersection(negation_words))

            negation_counts.loc['negation', class_] += negation_count
            negation_counts.loc['total', class_] += len(tokens)

    negation_rate = negation_counts.loc['negation',
                                        :] / negation_counts.loc['total', :]
    negation_rate *= 100
    idx = np.arange(2*len(negation_counts.columns), step=2)
    plt.figure()
    plt.bar(idx, negation_rate, edgecolor='black')
    plt.xticks(idx, negation_counts.columns)
    plt.xlabel('Classes')
    plt.ylabel('Rate of negation words (%)')
    plt.show()


def main():
    data_path = 'hate-speech-dataset'
    lexicon_path = 'lexicon'
    # visualize_class_balance(data_path)
    # visualize_tags(data_path)
    visualize_polarities(data_path, lexicon_path)
    visualize_negation(data_path, lexicon_path)


if __name__ == "__main__":
    main()
