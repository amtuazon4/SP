# File for preprocessing

import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

lemmatizer = WordNetLemmatizer()

# Function for reading the dataset
def read_json(filename):
    f = open(filename, "r")
    return json.load(f)

# Function for tokenizing a text
# Also lemmatizes it
def token_lem(text):
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(token).lower() for token in tokens]

# Function for removing duplicates
def rem_dup(arr):
    return list(dict.fromkeys(arr))

# Function for getting the input and output text of the json data
def get_inp_out(json_data):
    inp, out_class, out_num = [], [], []
    num = 0
    for intent in json_data["intents"]:
        for text in intent["text"]:
            inp.append(text)
            out_class.append(intent["intent"])
            out_num.append(num)
        num += 1
    return inp, out_class, out_num

# Function for generating the bag of words
def gen_BOW(inp):
    bow = []
    for text in inp:
        for token in rem_dup(token_lem(text)):
            if(token not in bow): bow.append(token)
    return sorted(bow)

# Function for preprocessing the data using Bag of Words
def BOW_preprocess(inp, bow):
    output = []
    for text in inp:
        inp_row = rem_dup(token_lem(text))
        temp = np.array([1 if(x in inp_row) else 0 for x in bow])
        output.append(temp)
    return output

# Function for preprocessing the data using TF-IDF
def TFIDF_preprocess(inp, bow):
    vectorizer = TfidfVectorizer(vocabulary=bow, tokenizer=token_lem)
    tfidf_matrix = vectorizer.fit_transform(inp)
    df = pd.DataFrame(tfidf_matrix.todense(), columns=vectorizer.get_feature_names_out())
    output = [np.array(row) for index, row in df.iterrows()]
    return output

# Function for preprocessing the data using word2vec
def W2Vec_preprocess(inp):
    tokenized_data = [token_lem(text) for text in inp]
    w2vec_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
    output = []
    for data in tokenized_data:
        valid_tokens = [token for token in data if token in w2vec_model.wv]
        if(valid_tokens):
            output.append(np.mean([w2vec_model.wv[token] for token in valid_tokens], axis=0))
        else:
            output.append(np.zeros(w2vec_model.wv.vector_size))
    return output, w2vec_model


