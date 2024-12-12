from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Attention mechanism for temporal or channel focus
def attention_module(inputs):
    """Simple attention mechanism."""
    attention_weights = layers.GlobalAveragePooling1D()(inputs)
    attention_weights = layers.Dense(units=inputs.shape[-1], activation='softmax')(attention_weights)
    attention_weights = layers.Reshape((1, inputs.shape[-1]))(attention_weights)
    outputs = layers.Multiply()([inputs, attention_weights])
    return outputs

# Basic residual block for ResNet-1D
def residual_block_1d(x, filters, kernel_size, stride=1):
    shortcut = x

    # First convolution
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolution
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Adjust shortcut if dimensions differ
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add shortcut and pass through ReLU
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

# ResNet-1D model with attention mechanism and metadata input
def create_resnet1d_with_metadata(input_shape, num_classes, metadata_input_shape, block_depths=[2, 2, 2, 2], initial_filters=64):
    # Signal input branch
    signal_inputs = layers.Input(shape=input_shape, name="signal_input")
    x = layers.Conv1D(initial_filters, kernel_size=7, strides=2, padding='same', use_bias=False)(signal_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    filters = initial_filters
    for depth in block_depths:
        for i in range(depth):
            stride = 2 if i == 0 and filters != initial_filters else 1
            x = residual_block_1d(x, filters, kernel_size=3, stride=stride)
        filters *= 2

    # Apply attention mechanism
    x = attention_module(x)

    # Flatten signal branch output
    x = layers.GlobalAveragePooling1D()(x)

    # Metadata input branch
    metadata_inputs = layers.Input(shape=metadata_input_shape, name="metadata_input")
    y = layers.Dense(32, activation='relu')(metadata_inputs)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(16, activation='relu')(y)

    # Concatenate both branches
    combined = layers.Concatenate()([x, y])

    # Classification layers
    z = layers.Dense(512, activation='relu')(combined)
    z = layers.Dropout(0.5)(z)
    outputs = layers.Dense(num_classes, activation='softmax')(z)

    model = models.Model(inputs=[signal_inputs, metadata_inputs], outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Training and evaluation function
def train_and_evaluate_with_metadata(
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