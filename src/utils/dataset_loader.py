import os
import cv2
import numpy as np
from typing import Tuple, List

# Helper function to load images from a folder
def load_images_from_folder(folder_path: str, label: str = None) -> Tuple[List[np.ndarray], List[str]]:
    images = []
    labels = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        if filename.lower().endswith((".jpg", ".jpeg", ".png")): #extra formats incase we use diff datasets later
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
                if label is not None:
                    labels.append(label)
                else:
                    label_name = os.path.splitext(filename)[0].split("_")[0]
                    labels.append(label_name.upper())

    return images, labels

# Loads all training data from subfolders
def load_train_data(train_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []

    for label_name in sorted(os.listdir(train_dir)):
        label_path = os.path.join(train_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        print(f"Loading {label_name}...")
        imgs, labels = load_images_from_folder(label_path, label_name)
        X.extend(imgs)
        y.extend(labels)

    print(f"Loaded {len(X)} training images across {len(set(y))} classes.")
    return X, y

# Loads all test data from a single folder
def load_test_data(test_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    X_test, y_test = load_images_from_folder(test_dir)
    print(f"Loaded {len(X_test)} test images across {len(set(y_test))} classes.")
    return X_test, y_test

# Loads both training and testing datasets
def load_dataset(root_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dir = os.path.join(root_dir, "asl_alphabet_train") # Training data directory
    test_dir = os.path.join(root_dir, "asl_alphabet_test")

    X_train, y_train = load_train_data(train_dir)
    X_test, y_test = load_test_data(test_dir)

    print("Dataset loaded successfully!")
    return X_train, y_train, X_test, y_test
