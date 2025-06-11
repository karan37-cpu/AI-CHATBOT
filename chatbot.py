import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import random
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data (run only once)
nltk.download('punkt')
nltk.download('wordnet')

# Load intents
with open("aichatbot/intents.json") as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

# Preprocessing
corpus = []
tags = []
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern.lower())
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
        corpus.append(" ".join(lemmatized))
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Train model
model = LogisticRegression()
model.fit(X, tags)

# Chat function
def chatbot_response(user_input):
    tokens = nltk.word_tokenize(user_input.lower())
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
    cleaned_input = " ".join(lemmatized)
    vector = vectorizer.transform([cleaned_input])

    pred_tag = model.predict(vector)[0]
    return random.choice(responses[pred_tag])

# Run chatbot
print("🤖 Chatbot is running! Type 'quit' to exit.")
while True:
    inp = input("You: ")
    if inp.lower() == "quit":
        print("Bot: Goodbye!")
        break
    print("Bot:", chatbot_response(inp))
