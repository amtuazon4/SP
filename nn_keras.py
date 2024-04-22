import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Class for the Neural network
# implements a Feed forward Neural network
class Neural_net():
    # Initialize the Neural network
    def __init__(self, Inodes, Hlayers, Hnodes, Onodes):
        self.model = Sequential()
        self.model.add(Dense(Hnodes, activation="relu", input_dim=Inodes))
        for i in range(Hlayers-1):
            self.model.add(Dense(Hnodes, activation="relu"))
        self.model.add(Dense(Onodes, activation="softmax"))
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["mse"])
    
    # Trains the neural network
    # returns the history of the training
    def train(self, x, y, epoch, b_size, verbose=0):
        return self.model.fit(x, y, epochs=epoch, batch_size=b_size, verbose=0)
    
    # Evaluates the input data provided using the model
    def evaluate(self, input_data, labels):
        return model.evaluate(input_data, labels)

    

    



    