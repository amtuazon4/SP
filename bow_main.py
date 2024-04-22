# File used for testing the chatbot using Bag-of-Words
from preprocess import *
from nn_keras import *

# Read the dataset
json_data = read_json("init_data.json")
inp_text, out_name, out_num = get_inp_out(json_data)

# Get the bag of words of the text
bow = gen_BOW(inp_text)
bow_len = (len(bow))

# Preprocesses the text using Bag of Words
inp_bow = BOW_preprocess(inp_text, bow)

# Create the neural network
nn = Neural_net(bow_len, 1, 272, 1)

