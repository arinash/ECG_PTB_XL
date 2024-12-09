from tensorflow.keras import layers, models

#ResNet with attention
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

# ResNet-1D model with attention mechanism and block depths
def create_resnet1d_with_attention(input_shape, num_classes, block_depths=[2, 2, 2, 2], initial_filters=64):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(initial_filters, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
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

    # Classification layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model