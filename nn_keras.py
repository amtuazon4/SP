import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

def measure_acc(pred_out, true_out, val_index, ref):
    inp, out, inp_proc, out_proc = ref
    count = 0
    for i, val_i in enumerate(val_index):
        if(pred_out[i] in out[inp.index(inp_proc[val_i])]): count += 1
    return count/len(val_index)    

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
    def train(self, trainX, trainY, validX, validY, epoch, b_size):
        # trainY = to_categorical(trainY)
        # validY = to_categorical(validY)
        checkpoint = ModelCheckpoint('model_temp.keras', monitor='val_mse', mode='min', save_best_only=True, verbose=1)
        return self.model.fit(trainX, trainY, validation_data=(validX, validY), epochs=epoch, batch_size=b_size, callbacks=[checkpoint])

    # Evaluates the input data provided using the model
    def evaluate(self, input_data, labels):
        return model.evaluate(input_data, labels)

    def kfold_eval(self, inp_data, out_data, ref):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        kfold_indices = kfold.split(inp_data, out_data)
        acc = []
        f1_scores = []
        for train_index, val_index in kfold_indices:
            self.init_model()
            inp_train = np.array([inp_data[i] for i in train_index])
            out_train = np.array([out_data[i] for i in train_index])
            inp_valid = np.array([inp_data[i] for i in val_index])
            out_valid = np.array([out_data[i] for i in val_index])
            hist = self.train(inp_train, out_train, inp_valid, out_valid, 10, 32)
            temp_model = load_model("model_temp.keras")
            predictions = temp_model.predict(inp_valid)

            pred_out = [np.argmax(x) for x in predictions]
            true_out = [np.argmax(x) for x in out_valid]
            
            acc.append(measure_acc(pred_out, true_out, val_index, ref))
            f1_scores.append(f1_score(true_out, pred_out, average="micro"))

        acc = np.array(acc)
        f1_scores = np.array(f1_scores)
        return np.mean(acc), np.mean(f1_scores)
        



    



    