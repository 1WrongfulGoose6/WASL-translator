import csv, os, datetime
import pickle
import numpy as np
# Model related imports
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

tf.random.set_seed(42)
np.random.seed(42)


def save_run_metrics(run, accuracy, precision, recall, f1, FAR, roc_area, auc, csv_path="model_results.csv"):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Header on first creation
        if not file_exists:
            writer.writerow([
                "timestamp", "run",
                "accuracy", "precision", "recall",
                "f1", "false_acceptance_rate",
                "roc_area", "auc"
            ])

        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run,
            accuracy,
            precision,
            recall,
            f1,
            FAR,
            roc_area,
            auc
        ])


# Root directory containing all runs
root_dir = r"E:\Ashwin\importantFiles\Programming Projects\Runs"

# List of preprocessing folders
run_folders = [
    "JustGaussianBlur",
    "JustOtsu",
    "JustRobert",
    "JustGrayscale",
    "JustPrewitt",
    "JustSobel"
]

for run in run_folders:
    tf.keras.backend.clear_session()
    run_path = os.path.join(root_dir, run)
    print(f"\n========== RUNNING: {run} ==========")

    # Load processed data
    X_train_proc = np.load(os.path.join(run_path, "X_train_processed.npy"))
    y_train_enc  = np.load(os.path.join(run_path, "y_train_encoded.npy"))
    X_test_proc  = np.load(os.path.join(run_path, "X_test_processed.npy"))
    y_test_enc   = np.load(os.path.join(run_path, "y_test_encoded.npy"))

    with open(os.path.join(run_path, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_proc, y_train_enc, test_size=0.1, stratify=y_train_enc, random_state=42
    )
    # Create results folder inside this run folder
    results_dir = os.path.join(run_path, "results")
    os.makedirs(results_dir, exist_ok=True)


    # One-hot encode
    num_classes = len(label_encoder.classes_)
    y_train = to_categorical(y_train, num_classes)
    y_val   = to_categorical(y_val, num_classes)
    y_test  = to_categorical(y_test_enc, num_classes)

    # Build model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, 
                               restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(results_dir, f"best_model_{run}.keras"),
        monitor='val_accuracy', save_best_only=True)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Predictions
    y_prob = model.predict(X_test_proc)
    y_pred = np.argmax(y_prob, axis=1)

    # Metrics
    accuracy = accuracy_score(y_test_enc, y_pred)
    precision = precision_score(y_test_enc, y_pred, average="macro")
    recall = recall_score(y_test_enc, y_pred, average="macro")
    f1 = f1_score(y_test_enc, y_pred, average="macro")

    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except Exception as e:
        print(f"AUC failed: {e}")
    auc = np.nan

    roc_area = auc
    
    cm = confusion_matrix(y_test_enc, y_pred)

    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
    FAR = np.mean(FP / (FP + TN + 1e-6))

    save_run_metrics(run, accuracy, precision, recall, f1, FAR, roc_area, auc)
    print(f"Saved metrics for {run}")
    with open(f"{results_dir}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"FAR: {FAR:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
    


    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap="viridis")
    plt.title(f"Confusion Matrix - {run}")
    plt.savefig(f"{results_dir}/{run}_confusion_matrix.png", dpi=300)

    plt.close()

    # Plot training curves
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title(f"Accuracy Curve - {run}")
    plt.legend()
    plt.savefig(f"{results_dir}/{run}_accuracy_curve.png")

    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"Loss Curve - {run}")
    plt.legend()
    plt.savefig(f"{results_dir}/{run}_loss_curve.png")

    plt.close()


