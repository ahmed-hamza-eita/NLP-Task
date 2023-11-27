import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def load_data(fake_csv, true_csv):
    fake_data = pd.read_csv(fake_csv)
    true_data = pd.read_csv(true_csv)
    data_merge = pd.concat([fake_data, true_data], axis=0)
    data_merge = data_merge.sample(frac=0.01, random_state=42)
    return data_merge


def drop_columns(data):
    return data.drop(['title', 'subject', 'date'], axis=1)


def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    stemmer = PorterStemmer()
    tokenized_text = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_text_no_stopwords = [word for word in tokenized_text if word.lower() not in stop_words]
    filtered_text_stemmer = [stemmer.stem(word) for word in filtered_text_no_stopwords]
    text = ' '.join(filtered_text_stemmer)
    return text


def apply_preprocessing(data):
    data['text'] = data['text'].apply(preprocess_text)
    return data


def save_to_file(data, filename):
    with open(filename, 'w') as file:
        for text in data['text']:
            file.write(str(text) + '\n')
