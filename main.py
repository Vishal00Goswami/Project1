import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
import os

# Initialize Porter Stemmer
ps = PorterStemmer()

# Download NLTK stopwords
nltk.download('stopwords')

# Text preprocessing function
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.compile("<.*?>").sub('', text)
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Apply stemming
    text = " ".join(ps.stem(word) for word in text.split())
    return text

# Load model and vectorizer
MODEL_PATH = r'E:\Muj_2024_Assigment\SentimentAnalyzer\pythonProject\model.pkl'
VECTORIZER_PATH = r'E:\Muj_2024_Assigment\SentimentAnalyzer\pythonProject\vectorizer.pkl'

# Check for file existence and load
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error(f"Model or vectorizer file not found in the specified path. "
             f"Please ensure these files are present:\n{MODEL_PATH}\n{VECTORIZER_PATH}")
else:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        cv = pickle.load(vectorizer_file)

# Streamlit UI
st.title("Sentiment Analyzer")
st.markdown("### Analyze the sentiment of your review")

input_review = st.text_area("Enter the review:", placeholder="Type your review here...")

if st.button("Predict"):
    if not input_review.strip():
        st.warning("Please enter a valid review.")
    else:
        # Preprocess the input
        transformed_review = transform_text(input_review)

        # Vectorize the input
        try:
            vector_input = cv.transform([transformed_review])  # Wrap in a list
            # Predict sentiment
            result = model.predict(vector_input)

            # Display the result
            if result[0] == 0:
                st.success("### Sentiment: Negative")
            else:
                st.success("### Sentiment: Positive")
        except Exception as e:
            st.error(f"An error occurred while processing: {e}")
