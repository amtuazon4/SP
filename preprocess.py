# File for preprocessing

import nltk
import json
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Function for reading the dataset
def read_json(filename):
    f = open(filename, "r")
    return json.load(f)

# Function for tokenizing strings
def tokenize(text):
    return nltk.word_tokenize(text)

# Function for stemming/lemmatization of a token
def lem(word):
    return stemmer.stem(word.lower())

# Function for removing duplicates
def rem_dup(arr):
    return list(dict.fromkeys(arr))

# Function for generating the bag of words
def gen_BOW(json_data):
    bow = []
    for intent in json_data["intents"]:
        for text in intent["text"]:
            for token in rem_dup([lem(x) for x in tokenize(text)]):
                if(token not in bow): bow.append(token)
    return sorted(bow)

# Function for preprocessing the data using Bag of Words
def BOW_preprocess(json_data, bow):
    inp = []
    out = []
    for intent in json_data["intents"]:
        for text in intent["text"]:
            inp_row = rem_dup([lem(x) for x in tokenize(text)])
            temp = [1 if(x in inp_row) else 0 for x in bow]
            inp.append(temp)
            out.append(intent["intent"])
    return inp, out

# Function for preprocessing the data using TF-IDF
def TFIDF_preprocess(json_data):
    vectorizer = TfidfVectorizer()
    doc = []
    out = []
    for intent in json_data["intents"]:
        for text in intent["text"]:
            doc.append(text)
            out.append(intent["intent"])
    tfidf_matrix = vectorizer.fit_transform(doc)
    df = pd.DataFrame(tfidf_matrix.todense(), columns=vectorizer.get_feature_names_out())
    inp = [list(row) for index, row in df.iterrows()]
    return inp, out


    
        





# Testing Area
temp = read_json("init_data.json")
x, y = TFIDF_preprocess(temp)

# text = "I AM ATOMIC."
# tokens = tokenize(text)
# temp = fix_tokens(tokens)
