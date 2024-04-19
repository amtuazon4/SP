# File for preprocessing
import nltk
import json
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

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

# Function for generating the bag of words
def gen_BOW(json_data):
    bow = []
    for intent in json_data["intents"]:
        for text in intent["text"]:
            for token in [lem(x) for x in tokenize(text)]:
                if(token not in bow): bow.append(token)
    return bow





# Testing Area
temp = read_json("init_data.json")
gen_BOW(temp)

# text = "I AM ATOMIC."
# tokens = tokenize(text)
# temp = fix_tokens(tokens)
