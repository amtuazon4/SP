# File for preprocessing

import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
    inp, out = [], []
    for intent in json_data["intents"]:
        for text in intent["text"]:
            inp.append(text)
            out.append(intent["intent"])
    return inp, out

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
        temp = [1 if(x in inp_row) else 0 for x in bow]
        output.append(temp)
    return output

# Function for preprocessing the data using TF-IDF
def TFIDF_preprocess(inp, bow):
    vectorizer = TfidfVectorizer(vocabulary=bow, tokenizer=token_lem)
    tfidf_matrix = vectorizer.fit_transform(inp)
    df = pd.DataFrame(tfidf_matrix.todense(), columns=vectorizer.get_feature_names_out())
    output = [list(row) for index, row in df.iterrows()]
    return output

def W2Vec_preprocess(json_data):
    pass

    
        





# Testing Area

json_data = read_json("init_data.json")
x, y = get_inp_out(json_data)
bow = gen_BOW(x)
TFIDF_preprocess(x, bow)


# text = "I AM ATOMIC."
# tokens = tokenize(text)
# temp = fix_tokens(tokens)
