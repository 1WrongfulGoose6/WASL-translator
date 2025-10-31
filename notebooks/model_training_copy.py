#!/usr/bin/env python
# coding: utf-8

params = globals().get("params", {})
run_name = params.get("run_name")

data_dir = f"../data/processed/{run_name}"

X_train = np.load(f"{data_dir}/X_train.npy")
X_test  = np.load(f"{data_dir}/X_test.npy")
y_train = np.load(f"{data_dir}/y_train.npy")
y_test  = np.load(f"{data_dir}/y_test.npy")

print(f"Training model for dataset: {run_name}")


# #### 1. Loading Processed Data

# In[ ]:


import numpy as np
import os

# Path relative to notebook directory
# base_path = "../data/processed"

# X_train = np.load(os.path.join(base_path, "X_train_processed.npy"))
# X_test = np.load(os.path.join(base_path, "X_test_processed.npy"))
# y_train = np.load(os.path.join(base_path, "y_train_encoded.npy"))
# y_test = np.load(os.path.join(base_path, "y_test_encoded.npy"))

# print(X_train.shape, y_train.shape)


# #### 2. Initial Setup

# ##### 2.1 Split training/validation sets

# In[12]:


from sklearn.model_selection import train_test_split
# split the data into 10% validation set and 90% training set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)


# In[13]:


from tensorflow.keras.utils import to_categorical # Pylance might still warn, but it's correct
# One hot encoding to handle categorical formatting
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)


# ##### 2.2 Building CNN model

# In[14]:


from tensorflow.keras import layers, models

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

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# #### 3. Training Model

# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    "model_checkpoint.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=[checkpoint]
)


# ### 4. Evaluation

# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:


model.save("../models/asl_cnn_model.h5")


# In[ ]:




with open(f"../results/{run_name}_metrics.txt", "w") as f:
    f.write(f"Accuracy: {max(history.history['val_accuracy'])}\n")
    f.write(f"Loss: {min(history.history['val_loss'])}\n")
