# File used for testing the chatbot using Word2Vec
from preprocess import *
from nn_keras import *
from keras.utils import to_categorical
import time
import numpy as np

# Read the dataset
json_data = read_json("init_data2.json")
inp, out, resp, inp_proc, out_proc = get_inp_resp(json_data)
out_proc2, lb = preprocess_out(out_proc)

start_time = time.time()
# Preprocesses the text using Bag of Words
inp_w2vec, w2vec_model = W2Vec_preprocess(inp_proc)
preprocessing_time = time.time() - start_time


# Create the neural network
nn = Neural_net(len(inp_w2vec[0]), 1, 544, len(out_proc2[0]))

# Perform K-folds validation
epochs = 500
batch_size = 32
acc, f1 , train_time = nn.kfold_eval(inp_w2vec, out_proc2, "w2vec",(inp, out, resp, inp_proc, out_proc), epochs, batch_size)
fp = open("w2vec_models/w2vec_results.txt", "w")
fp.write(f"accuracy: {acc}\n")
fp.write(f"f1-score: {f1}\n")
fp.write(f"preprocessing_time: {preprocessing_time}\n")
fp.write(f"preprocessing_time + training_time: {preprocessing_time + train_time}\n")
fp.write(f"epochs, batch_size = {epochs}, {batch_size}")
fp.close()