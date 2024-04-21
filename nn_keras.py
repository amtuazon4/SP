import numpy as np
from keras.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense


class Neural_net():
    def __init__(self, Inodes, Hlayers, Hnodes, Onodes):
        self.model = Sequential()
        self.model.add(Dense(Hnodes, activation="relu", input_dim=Inodes))
        for i in range(Hlayers-1):
            self.model.add(Dense(Hnodes, activation="relu"))
        self.model.add(Dense(Onodes, activation="softmax"))



    