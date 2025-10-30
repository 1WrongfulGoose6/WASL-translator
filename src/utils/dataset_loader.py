import os
import cv2
import numpy as np
from typing import Tuple, List

# Helper function to load images from a folder
def load_images_from_folder(folder_path: str, label: str = None, limit: int = None):
    images, labels = [], []

    for i, filename in enumerate(os.listdir(folder_path)):
        if limit and i >= limit:
            break  # stop after hitting limit
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
                if label is not None:
                    labels.append(label)
                else:
                    label_name = os.path.splitext(filename)[0].split("_")[0]
                    labels.append(label_name.lower())

    return images, labels


# Loads all training data from subfolders
def load_train_data(train_dir: str, debug: bool = False):
    X, y = [], []
    limit = 5 if debug else None  # only load 20 per class when debugging
    for label_folder in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, label_folder)
        if os.path.isdir(folder_path):
            print(f"Loading {label_folder}...")
            imgs, labels = load_images_from_folder(folder_path, label_folder.lower(), limit=limit)
            X.extend(imgs)
            y.extend(labels)
    print(f"Loaded {len(X)} training images across {len(set(y))} classes.")
    return X, y


# Loads all test data from a single folder
def load_test_data(test_dir: str, debug: bool = False):
    limit = 5 if debug else None
    X, y = load_images_from_folder(test_dir, limit=limit)
    print(f"Loaded {len(X)} test images across {len(set(y))} labels.")
    return X, y


# Loads both training and testing datasets
def load_dataset(root_dir: str, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dir = os.path.join(root_dir, "asl_alphabet_train") # Training data directory
    test_dir = os.path.join(root_dir, "asl_alphabet_test")

    X_train, y_train = load_train_data(train_dir)
    X_test, y_test = load_test_data(test_dir)

    print("Loaded dataset_loader from:", __file__)

    print("Train label set:", sorted(set(y_train)))
    print("Test label set:", sorted(set(y_test)))
    print("Unseen in test:", set(y_test) - set(y_train))

    return X_train, y_train, X_test, y_test
