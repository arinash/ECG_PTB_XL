import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Training and evaluation function
def train_and_evaluate(
    model,
    X_train, y_train, X_val, y_val, X_test, y_test,
    metadata_train, metadata_val, metadata_test,
    model_save_path, num_epochs=20):

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        [X_train, metadata_train], y_train,
        epochs=num_epochs,
        batch_size=32,
        validation_data=([X_val, metadata_val], y_val),
        callbacks=[early_stopping]
    )

    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.show()

    # Evaluate on the test set
    test_loss, test_acc = model.evaluate([X_test, metadata_test], y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Save the trained model
    print(f"Saving the trained model to '{model_save_path}'...")
    model.save(model_save_path)
    print("Model saved successfully!")

    # Make predictions on the test set
    y_pred = model.predict([X_test, metadata_test])
    print("Predictions (likelihoods) for the first 5 test samples:")
    print(y_pred[:5])