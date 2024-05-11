import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Generate synthetic dataset with class imbalance
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, weights=[0.1, 0.3, 0.6], random_state=42)

# Split the dataset into training and validation sets
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

# Convert class weights to a dictionary
class_weight_dict = dict(enumerate(class_weights))

# Print class weights
print("Class Weights:")
for class_idx, weight in class_weight_dict.items():
    print(f"Class {class_idx}: {weight:.4f}")


print("y")