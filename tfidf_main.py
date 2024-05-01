# File used for testing the chatbot using TF-IDF
from preprocess import *
from nn_keras import *
from keras.utils import to_categorical
import time
import numpy as np

# Read the dataset and preprocesses the dataset
json_data = read_json("init_data2.json")
inp, out, resp, inp_proc, out_proc = get_inp_resp(json_data)
out_proc2, lb = preprocess_out(out_proc)

# Get the bag of words of the text
bow = gen_BOW(inp)

# Preprocesses the text using Bag of Words
start_time = time.time()
tfidf_bow = TFIDF_preprocess(inp_proc, bow)
preprocessing_time = time.time() - start_time

# Create the neural network
nn = Neural_net(len(bow), 1, 272, len(out_proc2[0]))

# Perform K-folds validation
epochs = 10
batch_size = 32
acc, f1 , train_time = nn.kfold_eval(tfidf_bow, out_proc2, "tfidf",(inp, out, resp, inp_proc, out_proc), epochs, batch_size)
fp = open("tfidf_models/tfidf_results.txt", "w")
fp.write(f"accuracy: {acc}\n")
fp.write(f"f1-score: {f1}\n")
fp.write(f"preprocessing_time: {preprocessing_time}\n")
fp.write(f"preprocessing_time + training_time: {preprocessing_time + train_time}\n")
fp.write(f"epochs, batch_size = {epochs}, {batch_size}")
fp.close()


