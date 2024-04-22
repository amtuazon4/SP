# File used for testing the chatbot using TF-IDF
from preprocess import *
from nn_keras import *
from keras.utils import to_categorical
import time
import numpy as np

# Read the dataset
json_data = read_json("init_data.json")
inp_text, out_name, out_num = get_inp_out(json_data)

# Get the bag of words of the text
bow = gen_BOW(inp_text)


start_time = time.time()
# Preprocesses the text using Bag of Words
inp_tfidf = TFIDF_preprocess(inp_text, bow)
end_time = time.time()

# Create the neural network
nn = Neural_net(len(bow), 1, 272, 39)

# Perform K-folds validation
val = nn.kfold_eval(inp_tfidf, out_num)
print(f"Processing Time: {end_time - start_time} seconds.")
print(f"Accuracy: {val}")
