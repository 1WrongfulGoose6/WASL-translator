
from dataset_loader import load_dataset

X_train, y_train, X_test, y_test = load_dataset("./data/raw")

print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("Classes:", sorted(set(y_train)))
