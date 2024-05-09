import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import matplotlib.pyplot as plt
import os
import time

def acc_per_label(pred_out_binary, true_out):
    accuracy_per_label = []
    for i in range(true_out.shape[1]):
        TP = np.sum((pred_out_binary[:,i]==1) & (true_out[:,i]==1))
        TN = np.sum((pred_out_binary[:,i]==0) & (true_out[:,i]==0))
        FP = np.sum((pred_out_binary[:,i]==1) & (true_out[:,i]==0))
        FN = np.sum((pred_out_binary[:,i]==0) & (true_out[:,i]==1))
        
        label_acc = (TP+TN)/(TP+TN+FP+FN)
        accuracy_per_label.append(label_acc)
    return accuracy_per_label

def comp_class_freq(true_out):
    return np.sum(true_out, axis=0)

# Function for measuring the accuracy of the model
def measure_acc(pred_out, true_out, val_index, ref):
    inp, out, resp = ref
    pred_out1 = np.round(np.array(pred_out) >= 0.5)
    accuracy_per_label = acc_per_label(pred_out1, true_out)
    class_freq = comp_class_freq(true_out)
    return np.sum(accuracy_per_label * class_freq)/ np.sum(class_freq)
    

# Function for measuring the f1-score of the model
def measure_f1score(pred_out, true_out, val_index, ref):
    inp, out, resp = ref
    pred_out1 = np.array(pred_out) >= 0.5
    pred_out2 = pred_out1.astype(int)
    return f1_score(true_out, pred_out2, average="weighted", zero_division=0)

# Class for the Neural network
# implements a Feed forward Neural network
class Neural_net():
    # Initialize the Neural network
    def __init__(self, Inodes, Hlayers, Hnodes, Onodes):
        self.Inodes = Inodes
        self.Hlayers = Hlayers
        self.Hnodes = Hnodes
        self.Onodes = Onodes
        self.init_model()
    
    # initialize neural network model
    def init_model(self):
        self.model = Sequential()
        self.model.add(Input(shape=(self.Inodes,)))
        for i in range(self.Hlayers):
            self.model.add(Dense(self.Hnodes, activation="relu"))
        self.model.add(Dense(self.Onodes, activation="sigmoid"))
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["mse"])
    
    # Trains the neural network
    # returns the history of the training
    def train(self, trainX, trainY, validX, validY, epoch, b_size, fname):
        # trainY = to_categorical(trainY)
        # validY = to_categorical(validY)
        checkpoint = ModelCheckpoint(fname, monitor='val_mse', mode='min', save_best_only=True, verbose=1)
        return self.model.fit(trainX, trainY, validation_data=(validX, validY), epochs=epoch, batch_size=b_size, callbacks=[checkpoint])

    # Evaluates the input data provided using the model
    def evaluate(self, input_data, labels):
        return model.evaluate(input_data, labels)

    # Performs kfold cross validation
    def kfold_eval(self, inp_data, out_data, emb_type, ref, epoch, batch_size):
        self.reset_kfold_indices(emb_type)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        kfold_indices = kfold.split(inp_data, out_data)
        acc = []
        f1_scores = []
        train_times = []
        knum = 1
        for train_index, val_index in kfold_indices:
            # Reinitialize the neural network
            self.init_model()

            # Gets the 80% and 20% portions of the datasets
            inp_train = np.array([inp_data[i] for i in train_index])
            out_train = np.array([out_data[i] for i in train_index])
            inp_valid = np.array([inp_data[i] for i in val_index])
            out_valid = np.array([out_data[i] for i in val_index])

            # Train the Neural Network
            start_time = time.time()
            hist = self.train(inp_train, out_train, inp_valid, out_valid, epoch, batch_size, f"{emb_type}_models/{emb_type}_{knum}.keras")
            end_time = time.time()
            train_times.append(end_time-start_time)
        
            # Evaluate the accuracy and f1_score
            self.export_fig(hist, f"{emb_type}_{knum}", emb_type, knum)

            # save the training and validation indices
            self.export_kfold_indices(emb_type, train_index, val_index)

            temp_model = load_model(f"{emb_type}_models/{emb_type}_{knum}.keras")
            predictions = temp_model.predict(inp_valid)
            acc.append(measure_acc(predictions, out_valid, val_index, ref))
            f1_scores.append(measure_f1score(predictions, out_valid, val_index, ref))
            knum += 1
        
        # Takes the mean value of the accuracies, f1-socres, and training times of the splits
        acc = np.array(acc)
        f1_scores = np.array(f1_scores)
        train_times = np.array(train_times)
        return np.mean(acc), np.mean(f1_scores), np.mean(train_times)

    # Exports the line graph of the training error and validation error per epoch
    def export_fig(self, hist, filename, emb_type, knum):
        orig_dir = os.getcwd()
        os.chdir(orig_dir + f"\{emb_type}_models")
        plt.cla()
        train_loss = hist.history["loss"]
        val_loss = hist.history["val_loss"]
        epochs = range(1, len(train_loss)+1)
        plt.plot(epochs, train_loss, 'b', label='Training error')
        plt.plot(epochs, val_loss, 'r', label='Validation error')
        plt.title(f"{emb_type} Training and Validation Errors at fold {knum}")
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig(filename)
        os.chdir(orig_dir)

    def export_kfold_indices(self, emb_type, train_index, valid_index):
        orig_dir = os.getcwd()
        os.chdir(orig_dir + f"\{emb_type}_models")
        fp = open("kfold_indices.txt", "a")
        fp.write("".join([f"{x} " for x in train_index] + ["\n"]))
        fp.write("".join([f"{x} " for x in valid_index] + ["\n"]))
        fp.close()
        os.chdir(orig_dir)

    def reset_kfold_indices(self, emb_type):
        orig_dir = os.getcwd()
        if(os.path.exists(orig_dir + f"/{emb_type}_models")): 
            os.chdir(orig_dir + f"\{emb_type}_models")
            open("kfold_indices.txt", "w").close()
            os.chdir(orig_dir)


    



    