# File used for testing the chatbot using Word2Vec
from preprocess import *
from nn_keras import *
from keras.utils import to_categorical
import time
import numpy as np

# Read the dataset
json_data = read_json("init_data.json")
inp_text, out_name, out_num = get_inp_out(json_data)

start_time = time.time()
# Preprocesses the text using Bag of Words
inp_w2vec, w2vec_model = W2Vec_preprocess(inp_text)
end_time = time.time()

# Create the neural network
nn = Neural_net(100, 1, 272, 39)

# Perform K-folds validation
val = nn.kfold_eval(inp_w2vec, out_num)
print(f"Processing Time: {end_time - start_time} seconds.")
print(f"Accuracy: {val}")
