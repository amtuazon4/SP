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

# Function for generating the bag of words
def gen_BOW(json_data):
    bow = []
    for intent in json_data["intents"]:
        for text in intent["text"]:
            for token in rem_dup(token_lem(text)):
                if(token not in bow): bow.append(token)
    return sorted(bow)

# Function for preprocessing the data using Bag of Words
def BOW_preprocess(json_data, bow):
    inp = []
    out = []
    for intent in json_data["intents"]:
        for text in intent["text"]:
            inp_row = rem_dup(token_lem(text))
            temp = [1 if(x in inp_row) else 0 for x in bow]
            inp.append(temp)
            out.append(intent["intent"])
    return inp, out

# Function for preprocessing the data using TF-IDF
def TFIDF_preprocess(json_data, bow):
    vectorizer = TfidfVectorizer(vocabulary=bow, tokenizer=token_lem)
    doc = []
    out = []
    for intent in json_data["intents"]:
        for text in intent["text"]:
            doc.append(text)
            out.append(intent["intent"])
    tfidf_matrix = vectorizer.fit_transform(doc)
    df = pd.DataFrame(tfidf_matrix.todense(), columns=vectorizer.get_feature_names_out())
    print(df)
    inp = [list(row) for index, row in df.iterrows()]
    return inp, out

def W2Vec_preprocess(json_data):
    pass

    
        





# Testing Area
json_data = read_json("init_data.json")
bow = gen_BOW(json_data)
x,y = TFIDF_preprocess(json_data, bow)
print(x[1], y[1])

# text = "I AM ATOMIC."
# tokens = tokenize(text)
# temp = fix_tokens(tokens)
