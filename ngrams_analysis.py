import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import pandas as pd


def calc_ngrams(txt, n):
    ngrams_list = ngrams(word_tokenize(txt), n)
    return [gram for gram in ngrams_list]


def count_ngram(txt, n):
    ngrams_list = calc_ngrams(txt, n)
    fd = nltk.FreqDist(ngrams_list)
    for k, v in fd.items():
        print(k, v)


def calc_ngram_probabilities(txt, n):
    ngrams_list = list(ngrams(word_tokenize(txt), n))
    fd = nltk.FreqDist(ngrams_list)
    total_ngrams = sum(fd.values())
    ngram_probabilities = {ngram: freq / total_ngrams for ngram, freq in fd.items()}
    return ngram_probabilities


def analyze_ngrams(data, n):
    data['ngrams'] = data['text'].apply(lambda x: calc_ngrams(x, n))
    data['text'].apply(lambda x: count_ngram(x, n))
    data['ngram_probabilities'] = data['text'].apply(lambda x: calc_ngram_probabilities(x, n))
    return data


# def save_ngram_probabilities(data, sample_size, filename):
#     sample_data = data['ngram_probabilities'].sample(sample_size)
#     ngram_probabilities_df = pd.DataFrame(sample_data.tolist(), index=sample_data.index)
#     ngram_probabilities_df = ngram_probabilities_df.fillna(0)
#     ngram_probabilities_df.to_csv(filename)


def save_ngram_probabilities(data, sample_size, filename):
    # Check if sample_size is greater than the number of rows in the DataFrame
    if sample_size > len(data):
        sample_size = len(data)

    # Use replace=True if sample_size is greater than the number of rows
    replace = sample_size > len(data)

    sample_data = data['ngram_probabilities'].sample(sample_size, replace=replace)
    ngram_probabilities_df = pd.DataFrame(sample_data.tolist(), index=sample_data.index)
    ngram_probabilities_df = ngram_probabilities_df.fillna(0)
    ngram_probabilities_df.to_csv(filename)
