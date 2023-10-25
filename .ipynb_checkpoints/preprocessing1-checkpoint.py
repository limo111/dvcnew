import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data from the pickle files
with open('sentences.pkl', 'rb') as file:
    sentences = pickle.load(file)

with open('labels.pkl', 'rb') as file:
    labels = pickle.load(file)

# Split data into training and test sets
training_size = 20000
training_labels = np.array(labels[:training_size])
test_labels = np.array(labels[training_size:])

training_sentences = sentences[:training_size]
test_sentences = sentences[training_size:]

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=1000, oov_token='<00V>')
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)
train_pad = pad_sequences(sequences, maxlen=16, truncating='post', padding='post')

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_pad = pad_sequences(test_sequences, maxlen=16, truncating='post', padding='post')

# Save data to pickle files
with open('training_labels.pkl', 'wb') as file:
    pickle.dump(training_labels, file)

with open('test_labels.pkl', 'wb') as file:
    pickle.dump(test_labels, file)

with open('train_pad.pkl', 'wb') as file:
    pickle.dump(train_pad, file)

with open('test_pad.pkl', 'wb') as file:
    pickle.dump(test_pad, file)

print("Data has been saved to training_labels.pkl, test_labels.pkl, train_pad.pkl, and test_pad.pkl")
