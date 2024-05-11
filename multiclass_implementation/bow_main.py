# File used for testing the chatbot using Bag-of-Words
from preprocess import *
from nn_keras import *
from keras.utils import to_categorical
import time
import numpy as np

# Read the dataset and preprocesses the dataset
json_data = read_json("init_data2.json")
inp, out, tags = get_inp_out_tag(json_data)

# Preprocesses the output
out2 = preprocess_out(out)

# Get the bag of words of the text
bow = gen_BOW(inp)

# Preprocesses the text using Bag of Words
start_time = time.time()
inp_bow = BOW_preprocess(inp, bow)
preprocessing_time = time.time() - start_time

# Create the neural network
inp_nodes = len(bow)
hid_layers = 2
hid_nodes = 1024
out_nodes = len(out2[0])
nn = Neural_net(inp_nodes, hid_layers, hid_nodes, out_nodes)


# Perform K-folds validation
epochs = 500
batch_size = 32
acc, f1 , train_time = nn.kfold_eval(inp_bow, out2, "bow",(inp, out, tags), epochs, batch_size)

fp = open("bow_models/bow_results.txt", "w")
fp.write(f"accuracy: {acc}\n")
fp.write(f"f1-score: {f1}\n")
fp.write(f"preprocessing_time: {preprocessing_time}\n")
fp.write(f"train_time: {train_time}\n")
fp.write(f"preprocessing_time + training_time: {preprocessing_time + train_time}\n")
fp.write(f"epochs, batch_size = {epochs}, {batch_size}\n")
fp.write(f"inp_nodes, hid_layers, hid_nodes, out_nodes = {inp_nodes}, {hid_layers}, {hid_nodes}, {out_nodes}")
fp.close()