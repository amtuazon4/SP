import numpy as np
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense

# Sample text data
text_data = [
    "I enjoy playing football with my friends.",
    "Football is my favorite sport.",
    "Soccer is also a popular sport in many countries.",
    "I watch soccer matches on TV.",
]
labels = np.array([1, 1, 0, 0])  # Example labels (binary classification)

# Preprocess text data
def preprocess_text(text_data):
    # Tokenize text data
    tokenized_data = [text.split() for text in text_data]
    return tokenized_data

tokenized_data = preprocess_text(text_data)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

# Aggregate or pool word embeddings
def pool_word_embeddings(text_data, word_vectors):
    pooled_vectors = []
    for text in text_data:
        # Tokenize text
        tokens = text.split()
        # Filter tokens that are in the vocabulary of the Word2Vec model
        valid_tokens = [token for token in tokens if token in word_vectors]
        
        
        # Pool word embeddings using average pooling
        if valid_tokens:
            pooled_vector = np.mean([word_vectors[token] for token in valid_tokens], axis=0)
            pooled_vectors.append(pooled_vector)
        else:
            # Handle out-of-vocabulary words
            pooled_vectors.append(np.zeros(word_vectors.vector_size))
    return np.array(pooled_vectors)

# Generate pooled representations for input text data
pooled_input_data = pool_word_embeddings(text_data, word2vec_model.wv)

print(len(pooled_input_data))
# Define neural network architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=pooled_input_data.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse', "accuracy"])

# Train the model
history = model.fit(pooled_input_data, labels, epochs=100, validation_split=0.2)

# Extract training and validation loss values
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Display training and validation loss per epoch
for epoch in range(len(training_loss)):
    print(f"Epoch {epoch + 1}: Training Loss = {training_loss[epoch]}, Validation Loss = {validation_loss[epoch]}")

# Evaluate the model
loss, accuracy = model.evaluate(pooled_input_data, labels)
print("Accuracy:", accuracy)