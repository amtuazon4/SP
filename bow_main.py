# File used for testing the chatbot using Bag-of-Words
from preprocess import *
from nn_keras import *
from keras.utils import to_categorical

import numpy as np
from sklearn.model_selection import StratifiedKFold

# Read the dataset
json_data = read_json("init_data.json")
inp_text, out_name, out_num = get_inp_out(json_data)

# Get the bag of words of the text
bow = gen_BOW(inp_text)

# Preprocesses the text using Bag of Words
inp_bow = BOW_preprocess(inp_text, bow)

# Create the neural network
nn = Neural_net(len(bow), 1, 272, 39)

# Perform K-folds validation
nn.kfold_eval(inp_bow, out_num)
