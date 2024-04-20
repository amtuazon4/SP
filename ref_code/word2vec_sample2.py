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

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

# Generate word embeddings
word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}

# Input representation
input_text = "I love soccer"
input_tokens = word_tokenize(input_text.lower())
input_embedding = [word_embeddings[token] for token in input_tokens if token in word_embeddings]

print(input_embedding)
# Integration with chatbot model (example)
# Here, 'input_embedding' can be used as input features for your chatbot model
