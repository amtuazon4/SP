from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
    "i am atomic, 123"
]

# Step 3: Compute TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Optional: Get feature names (terms)
feature_names = vectorizer.get_feature_names_out()

# Optional: Convert TF-IDF matrix to dense format for visualization (not recommended for large matrices)
dense_tfidf_matrix = tfidf_matrix.todense()

# Optional: Visualize TF-IDF matrix
import pandas as pd
df = pd.DataFrame(dense_tfidf_matrix, columns=feature_names)