import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK data (only run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
def load_data():
    data = pd.read_csv('review.csv')
    return data

data = load_data()
st.write("Dataset Sample", data.head())

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
data['cleaned_review'] = data['review'].apply(preprocess_text)
st.write("Cleaned Dataset Sample", data[['review', 'cleaned_review']].head())

# Split dataset into training and testing
X = data['cleaned_review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test the model
y_pred = model.predict(X_test_vec)

# Display accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.text(classification_report(y_test, y_pred))

# Streamlit app for sentiment prediction
st.title("Sentiment Analysis App")
st.write("Enter a review below to classify it as Positive, Negative, or Neutral.")

# Input text from user
user_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess input text
        cleaned_input = preprocess_text(user_input)
        # Vectorize input text
        input_vec = vectorizer.transform([cleaned_input])
        # Predict sentiment
        prediction = model.predict(input_vec)[0]
        st.write(f"Predicted Sentiment: **{prediction.capitalize()}**")
    else:
        st.write("Please enter some text.")

