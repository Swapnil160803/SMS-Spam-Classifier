import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize preprocessing tools
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def transform_text(text):
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Tokenize the text
    tokens = nltk.word_tokenize(text)

    # 3. Remove punctuation and stopwords, then stem words
    cleaned_words = []

    for word in tokens:
        if word.isalnum() and word not in stop_words:
            stemmed_word = ps.stem(word)
            cleaned_words.append(stemmed_word)

    # 4. Return cleaned text
    return " ".join(cleaned_words)


# Load the saved vectorizer and model
try:
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

except FileNotFoundError as e:
    st.error(f"Required model file not found: {e.filename}")
    st.stop()

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# Streamlit application
st.title("SMS Spam Classifier")
st.write("Enter an SMS message to check whether it is spam.")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    if not input_sms.strip():
        st.warning("Please enter a message first.")

    else:
        # Preprocess the message
        transformed_sms = transform_text(input_sms)

        # Convert text into TF-IDF features
        vectorized_input = tfidf.transform([transformed_sms])

        # Predict the result
        result = model.predict(vectorized_input)[0]

        # Display the prediction
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")