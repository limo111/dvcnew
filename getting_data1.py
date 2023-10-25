import os
import wget
import json
import pickle

# Download the JSON data
!wget https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json

# Load JSON data into datastore
with open('sarcasm.json', 'r') as f:
    datastore = json.load(f)

# Extract sentences, labels, and url from the datastore
sentences = []
labels = []
url = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    url.append(item['article_link'])

# Save sentences, labels, and url to pickle files
with open('sentences.pkl', 'wb') as file:
    pickle.dump(sentences, file)

with open('labels.pkl', 'wb') as file:
    pickle.dump(labels, file)

with open('url.pkl', 'wb') as file:
    pickle.dump(url, file)

print("Data has been saved to sentences.pkl, labels.pkl, and url.pkl")
