import numpy as np
import os
import joblib
from resnet_model import create_resnet1d_with_attention
from train import *
from predict import *

if __name__ == '__main__':
    DATA_PATH = '../data/processed/'
    MODEL_PATH = '../models/'
    MODEL_NAME = 'resnet_attention_adam_relu_2222_05_32.h5'
    SAMPLING_RATE = 100

    X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(DATA_PATH, 'X_val.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(DATA_PATH, 'y_val.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(DATA_PATH, 'X_test.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'), allow_pickle=True)
    print("Data loaded successfully!")
    
    metadata_train = pd.read_csv(os.path.join(DATA_PATH, 'metadata_train.csv'), index_col=0)
    metadata_val = pd.read_csv(os.path.join(DATA_PATH, 'metadata_val.csv'), index_col=0)
    metadata_test = pd.read_csv(os.path.join(DATA_PATH, 'metadata_test.csv'), index_col=0)
    print("Metadata loaded successfully!")
    
    metadata_features = ['age', 'weight', 'height']
    metadata_train = metadata_train[metadata_features].fillna(0).values
    metadata_val = metadata_val[metadata_features].fillna(0).values
    metadata_test = metadata_test[metadata_features].fillna(0).values
    # Verify the shape
    print("Metadata train shape:", metadata_train.shape)
    print("Metadata val shape:", metadata_val.shape)
    print("Metadata test shape:", metadata_test.shape)

    le = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))
    label_classes = le.classes_
    print("Label encoder loaded successfully!")
    
    y_train = le.fit_transform(y_train)
    y_val = le.fit_transform(y_val)
    y_test = le.fit_transform(y_test)
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    print("Training and evaluating the ResNet model...")
    model_save_path = os.path.join(MODEL_PATH, MODEL_NAME)
    model = create_resnet1d_with_attention(X_train.shape[1:], num_classes)
    train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, model_save_path=model_save_path, num_epochs=2)

    # Load the model and label encoder
    print("Loading model and label encoder...")
    model = load_saved_model(os.path.join(MODEL_PATH, MODEL_NAME))
    #le = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))

    # Test the model
    print("Testing the model...")
    y_test_encoded, y_pred_encoded, y_pred_probs = test_model(model, X_test, metadata_test, y_test, le)

    # Evaluate model performance
    print("Evaluating model performance...")
    evaluate_model_performance(y_test_encoded, y_pred_encoded, y_pred_probs, label_classes)