
import streamlit as st
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------------------
# 1. Download NLTK resources
# -------------------------------
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    return set(stopwords.words("english"))

stop_words = download_nltk_resources()

# -------------------------------
# 2. Initialize Porter Stemmer
# -------------------------------
ps = PorterStemmer()

# -------------------------------
# 3. Text preprocessing function
# -------------------------------
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Remove punctuation and stopwords, then stem
    cleaned_words = []

    for word in tokens:
        if word.isalnum() and word not in stop_words:
            cleaned_words.append(ps.stem(word))

    return " ".join(cleaned_words)

# -------------------------------
# 4. Load model and vectorizer
# -------------------------------
@st.cache_resource
def load_model_files():
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    return tfidf, model

try:
    tfidf, model = load_model_files()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------------
# 5. Streamlit application
# -------------------------------
st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message to check whether it is spam or not.")

input_sms = st.text_area(
    "Enter your message",
    placeholder="Example: Congratulations! You won a free prize. Reply now!"
)

if st.button("Predict", type="primary"):

    if not input_sms.strip():
        st.warning("Please enter a message first.")

    else:
        try:
            # Preprocess message
            transformed_sms = transform_text(input_sms)

            # Convert text into TF-IDF features
            vectorized_input = tfidf.transform([transformed_sms])

            # Predict message category
            result = model.predict(vectorized_input)[0]

            # Show debugging information
            with st.expander("Prediction details"):
                st.write("Cleaned message:", transformed_sms)
                st.write("Prediction value:", result)
                st.write("Model classes:", model.classes_)

            # Display result
            if result == 1:
                st.error("🚨 Spam Message")
            else:
                st.success("✅ Not Spam")

            # Display probabilities when supported
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(vectorized_input)[0]

                st.write("Prediction confidence:")
                for label, probability in zip(
                    model.classes_, probabilities
                ):
                    st.write(f"Class {label}: {probability:.2%}")

        except LookupError:
            st.error(
                "NLTK tokenizer data is missing. "
                "Please check the app logs and NLTK downloads."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
