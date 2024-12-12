import numpy as np
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

# Load the trained model
def test_model(model, X_test, metadata_test, y_test, le, num_examples=5):
    """
    Test a loaded Keras model on test data.

    Parameters:
        model: Loaded Keras model.
        X_test (np.ndarray): Test signal data.
        metadata_test (np.ndarray): Test metadata.
        y_test (np.ndarray): True labels for the test data.
        le: Label encoder.

    Returns:
        y_test_encoded, y_pred_encoded, y_pred_probs
    """
    # Predict probabilities and labels
    y_pred_probs = model.predict([X_test, metadata_test])  # Correct input order
    y_pred_encoded = y_pred_probs.argmax(axis=1)

    # Decode true and predicted labels for better interpretability
    y_test_decoded = le.inverse_transform(y_test)
    y_pred_decoded = le.inverse_transform(y_pred_encoded)

    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred_encoded)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return y_test, y_pred_encoded, y_pred_probs


# Evaluate and visualize model performance
def evaluate_model_performance(y_test, y_pred_encoded, y_pred_probs, label_classes):
    """
    Evaluate and visualize model performance using various metrics.

    Parameters:
        y_test (np.ndarray): True labels (encoded).
        y_pred_encoded (np.ndarray): Predicted labels (encoded).
        y_pred_probs (np.ndarray): Predicted probabilities for each class.
        label_classes (list): List of class labels.
    """
    from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix, precision_score, recall_score, f1_score

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_encoded)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_classes,
        yticklabels=label_classes
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    if not label_classes or not isinstance(label_classes, list):
      raise ValueError("label_classes must be a non-empty list of class names.")

    print("Classification Metrics:")
    print(classification_report(y_test, y_pred_encoded, target_names=label_classes))


    # ROC Curve (One-vs-Rest)
    print("Plotting ROC Curve...")
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, class_label in enumerate(label_classes):
        y_test_binary = (y_test == i).astype(int)
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
    pr_auc = {}
    for i, class_label in enumerate(label_classes):
        y_test_binary = (y_test == i).astype(int)
        precision[class_label], recall[class_label], _ = precision_recall_curve(y_test_binary, y_pred_probs[:, i])
        pr_auc[class_label] = auc(recall[class_label], precision[class_label])
        plt.plot(recall[class_label], precision[class_label], label=f'{class_label} (AUC = {pr_auc[class_label]:.2f})')

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    # Print AUC for ROC and PR curves, Precision, Recall, and F1-Score
    print("Detailed Metrics:")
    for i, class_label in enumerate(label_classes):
        precision_value = precision_score(y_test, y_pred_encoded, average=None)[i]
        recall_value = recall_score(y_test, y_pred_encoded, average=None)[i]
        f1_value = f1_score(y_test, y_pred_encoded, average=None)[i]
        print(f"Class: {class_label}")
        print(f"  Precision: {precision_value:.2f}")
        print(f"  Recall: {recall_value:.2f}")
        print(f"  F1-Score: {f1_value:.2f}")
        print(f"  ROC AUC: {roc_auc[class_label]:.2f}")
        print(f"  PR AUC: {pr_auc[class_label]:.2f}")
        print()