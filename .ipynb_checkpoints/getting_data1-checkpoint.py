import os
import wget
import json
import pickle

import requests

url = "https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json"
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    with open("sarcasm.json", "wb") as f:
        f.write(response.content)
    print("File downloaded successfully.")
else:
    print("Failed to download the file.")

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
