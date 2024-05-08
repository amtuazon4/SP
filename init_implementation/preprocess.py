# File for preprocessing

import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

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
def get_inp_resp(json_data):
    # inp -> every input text in the dataset
    # out -> for every input text in inp, the corresponding group of responses (the indexes rather)
    # resp -> every response in the dataset, basis for the index in the out array
    # inp_proc -> input texts to be fed to the neural network
    # out_proc -> output classes for every input text
    inp, out = [], []
    resp = []
    inp_proc, out_proc = [], []

    count = 0
    for i, intent in enumerate(json_data["intents"]):
        inp += intent["patterns"]
        resp += intent["responses"]
        out += [list(range(count, count+len(intent["responses"]))) for x in range(len(intent["patterns"]))]
        for pat in intent["patterns"]:
            for num in list(range(count, count+len(intent["responses"]))):
                inp_proc.append(pat)
                out_proc.append(num)
        count += len(intent["responses"])
    return inp, out, resp, inp_proc, out_proc

def preprocess_out(out_proc):
    lb = LabelBinarizer()
    return lb.fit_transform(out_proc), lb

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
    return np.array(output)

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
    w2vec_model = Word2Vec(sentences=tokenized_data, vector_size=933, window=5, min_count=1, workers=4)
    output = []
    for data in tokenized_data:
        valid_tokens = [token for token in data if token in w2vec_model.wv]
        if(valid_tokens):
            output.append(np.mean([w2vec_model.wv[token] for token in valid_tokens], axis=0))
        else:
            output.append(np.zeros(w2vec_model.wv.vector_size))
    return output, w2vec_model

json_data = read_json("init_data2.json")
inp, out, resp, inp_proc, out_proc = get_inp_resp(json_data)
out_proc2, lb = preprocess_out(out_proc)

