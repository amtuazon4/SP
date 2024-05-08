import numpy as np

def compute_multilabel_accuracy(y_true, y_pred):
    """
    Compute accuracy for multilabel classification.

    Parameters:
    y_true : array-like, shape (n_samples, n_classes)
        True labels.
    y_pred : array-like, shape (n_samples, n_classes)
        Predicted labels.

    Returns:
    accuracy : float
        Accuracy score.
    """
    print(y_true)
    print(y_pred)

    # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
    y_pred_binary = np.array(y_pred) >= 0.5
    print(y_pred_binary)
    # Compute accuracy for each label
    accuracy_per_label = np.mean(y_pred_binary == y_true, axis=0)
    print(np.array(y_pred_binary == y_true))
    print(accuracy_per_label)
    # Average accuracy across all labels
    accuracy = np.mean(accuracy_per_label)

    return accuracy

# Example usage:
# Assuming y_true and y_pred are arrays containing true labels and predicted probabilities respectively
y_true = np.array([
    [1, 0, 1],  # Sample 1: Class 1 and Class 3
    [0, 1, 1],  # Sample 2: Class 2 and Class 3
    [1, 1, 0]   # Sample 3: Class 1 and Class 2
])

y_pred = np.array([
    [0.9, 0.1, 0.8],  # Sample 1: Probability of Class 1=0.9, Class 2=0.1, Class 3=0.8
    [0.2, 0.8, 0.7],  # Sample 2: Probability of Class 1=0.2, Class 2=0.8, Class 3=0.7
    [0.7, 0.6, 0.3]   # Sample 3: Probability of Class 1=0.7, Class 2=0.6, Class 3=0.3
])

accuracy = compute_multilabel_accuracy(y_true, y_pred)
print("Accuracy:", accuracy)