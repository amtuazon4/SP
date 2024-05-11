# File used for testing the chatbot using Word2Vec
from preprocess import *
from nn_keras import *
from keras.utils import to_categorical
import time
import numpy as np
import os

# Read the dataset
json_data = read_json("init_data2.json")
inp, out, tags = get_inp_out_tag(json_data)

# Preprocesses the output
out2 = preprocess_out(out)

# Preprocesses the text using word2vec
start_time = time.time()
inp_w2vec, w2vec_model = W2Vec_preprocess(inp)
preprocessing_time = time.time() - start_time

os.makedirs("w2vec_models", exist_ok=True)
w2vec_model_name = os.path.join("w2vec_models", "word2vec.bin")
w2vec_model.save(w2vec_model_name)

# Create the neural network
inp_nodes = len(inp_w2vec[0])
hid_layers = 3
hid_nodes = 1024
out_nodes = len(out2[0])
nn = Neural_net(inp_nodes, hid_layers, hid_nodes, out_nodes)

# Perform K-folds validation
epochs = 500
batch_size = 32
acc, f1 , train_time = nn.kfold_eval(inp_w2vec, out2, "w2vec",(inp, out, tags), epochs, batch_size)

fp = open("w2vec_models/w2vec_results.txt", "w")
fp.write(f"accuracy: {acc}\n")
fp.write(f"f1-score: {f1}\n")
fp.write(f"preprocessing_time: {preprocessing_time}\n")
fp.write(f"train_time: {train_time}\n")
fp.write(f"preprocessing_time + training_time: {preprocessing_time + train_time}\n")
fp.write(f"epochs, batch_size = {epochs}, {batch_size}\n")
fp.write(f"inp_nodes, hid_layers, hid_nodes, out_nodes = {inp_nodes}, {hid_layers}, {hid_nodes}, {out_nodes}")
fp.close()