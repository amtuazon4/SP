from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Sample text data
text_data = [
    "I enjoy playing football with my friends.",
    "Football is my favorite sport.",
    "Soccer is also a popular sport in many countries.",
    "I watch soccer matches on TV.",
]

# Tokenize the text data
tokenized_data = [word_tokenize(sentence.lower()) for sentence in text_data]

# Step 3: Train Word2Vec model
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

# Optional: Train for more epochs if needed
# model.train(tokenized_data, total_examples=len(tokenized_data), epochs=10)

# Step 4: Use the trained model
# Example: Find similar words
similar_words = model.wv.most_similar('football', topn=3)
print("Similar words to 'football':", similar_words)

# Example: Get word vector
word_vector = model.wv['football']
print("Vector representation of 'football':", word_vector)
