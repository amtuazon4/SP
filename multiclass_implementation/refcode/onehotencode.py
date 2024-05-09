from keras.utils import to_categorical

# Example integer labels
labels = [0, 1, 2, 1, 0]

# Perform one-hot encoding using to_categorical
one_hot_labels = to_categorical(labels)

print(one_hot_labels)