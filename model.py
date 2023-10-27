import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Conv1D, GlobalMaxPooling1D, LSTM
import numpy as np
import json
import pandas as pd

# Load training_labels from the pickle file
with open('training_labels.pkl', 'rb') as file:
    training_labels = pickle.load(file)

# Load test_labels from the pickle file
with open('test_labels.pkl', 'rb') as file:
    test_labels = pickle.load(file)

# Load train_pad from the pickle file
with open('train_pad.pkl', 'rb') as file:
    train_pad = pickle.load(file)

# Load test_pad from the pickle file
with open('test_pad.pkl', 'rb') as file:
    test_pad = pickle.load(file)

# Now training_labels, test_labels, train_pad, and test_pad are loaded back into your program
print("Data has been loaded from training_labels.pkl, test_labels.pkl, train_pad.pkl, and test_pad.pkl")

# List of models
models = [
    ("MLP", Sequential([
        Dense(64, input_shape=(16,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])),
    ("CNN", Sequential([
        Embedding(1000, 32, input_length=16),
        Conv1D(100, 3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])),
    ("LSTM", Sequential([
        Embedding(1000, 32, input_length=16),
        LSTM(100),
        Dense(1, activation='sigmoid')
    ])),
    ("GlobalAveragePooling1D", Sequential([
        Embedding(1000, 32, input_length=16),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ]))
]

# Training and evaluating each model

# Initialize an empty list to store model results
all_model_results = []

for name, model in models:
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_pad, training_labels, validation_data=(test_pad, test_labels), epochs=32, batch_size=32, verbose=0)
    loss, accuracy = model.evaluate(test_pad, test_labels, verbose=0)
    model_results = accuracy * 100
    all_model_results.append({"name": name, "accuracy": model_results})
    print(f"Model {name} - Test Accuracy: {accuracy * 100:.2f}%")

# Save all_model_results using pickle
with open('all_model_results.pkl', 'wb') as file:
    pickle.dump(all_model_results, file)

# Write all model results to a JSON file
with open("model_results.json", 'w') as outfile:
    json.dump(all_model_results, outfile)

print("Model results saved in model_results.json.")
