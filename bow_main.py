# File used for testing the chatbot using Bag-of-Words
from preprocess import *
from nn_keras import *
from keras.utils import to_categorical
import time
import numpy as np

# Read the dataset
json_data = read_json("init_data2.json")
inp, out, resp, inp_proc, out_proc = get_inp_resp(json_data)
out_proc2, lb = preprocess_out(out_proc)

# Get the bag of words of the text
bow = gen_BOW(inp)


start_time = time.time()
# Preprocesses the text using Bag of Words
inp_bow = BOW_preprocess(inp_proc, bow)
end_time = time.time()

# Create the neural network
nn = Neural_net(len(bow), 1, 272, len(out_proc2[0]))



# Perform K-folds validation
acc, f1 = nn.kfold_eval(inp_bow, out_proc2, (inp, out, inp_proc, out_proc))

print(acc, f1)
exit()
print(f"Processing Time: {end_time - start_time} seconds.")
print(f"Accuracy: {val}")
