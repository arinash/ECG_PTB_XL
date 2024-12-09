import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# Load the trained model
def load_saved_model(model_path):
    """
    Load a trained Keras model from a saved .h5 file.

    Parameters:
        model_path (str): Path to the saved .h5 model.

    Returns:
        model: Loaded Keras model.
    """
    model = load_model(model_path)
    print(f"Model loaded successfully from '{model_path}'!")
    return model

# Test the model
def test_model(model, X_test, y_test, le, num_examples=5):
    """
    Test a loaded Keras model on test data.

    Parameters:
        model: Loaded Keras model.
        X_test (np.ndarray): Test input data.
        y_test (np.ndarray): True labels for the test data.
        le: Label encoder.

    Returns:
        y_pred (np.ndarray): Predicted labels.
        y_test_decoded (np.ndarray): Decoded true labels.
    """
    # Encode the true labels
    y_test_encoded = le.transform(y_test)

    # Predict probabilities and labels
    y_pred_probs = model.predict(X_test)
    y_pred_encoded = y_pred_probs.argmax(axis=1)

    # Decode true and predicted labels for better interpretability
    y_test_decoded = le.inverse_transform(y_test_encoded)
    y_pred_decoded = le.inverse_transform(y_pred_encoded)

    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Display random predictions
    print("\nRandom Predictions vs True Labels:")
    indices = random.sample(range(len(X_test)), min(num_examples, len(X_test)))
    for idx in indices:
        print(f"Example {idx + 1}:")
        print(f"   True Label: {y_test_decoded[idx]}")
        print(f"   Predicted Label: {y_pred_decoded[idx]}")
        print(f"   Predicted Probabilities: {y_pred_probs[idx]}")
        print("-" * 40)

    return y_test_encoded, y_pred_encoded, y_pred_probs

# Evaluate and visualize model performance
def evaluate_model_performance(y_test_encoded, y_pred_encoded, y_pred_probs, le):
    """Evaluate and visualize model performance using various metrics."""
    label_classes = le.classes_

    # Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred_encoded)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_classes, yticklabels=label_classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # ROC Curve (One-vs-Rest)
    print("Plotting ROC Curve...")
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, class_label in enumerate(label_classes):
        y_test_binary = np.array([1 if label == i else 0 for label in y_test_encoded])
        fpr[class_label], tpr[class_label], _ = roc_curve(y_test_binary, y_pred_probs[:, i])
        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
        plt.plot(fpr[class_label], tpr[class_label], label=f'{class_label} (AUC = {roc_auc[class_label]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # Precision-Recall Curve (One-vs-Rest)
    print("Plotting Precision-Recall Curve...")
    precision = {}
    recall = {}
    for i, class_label in enumerate(label_classes):
        y_test_binary = np.array([1 if label == i else 0 for label in y_test_encoded])
        precision[class_label], recall[class_label], _ = precision_recall_curve(y_test_binary, y_pred_probs[:, i])
        plt.plot(recall[class_label], precision[class_label], label=class_label)

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()
