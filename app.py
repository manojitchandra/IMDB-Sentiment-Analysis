import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (safe to call multiple times)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


# Load stopwords
stop_words = set(stopwords.words("english"))
stop_words.remove("not")

# Load trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text cleaning function (must match training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)       # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)    # remove punctuation & numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Sentiment Analysis App")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and classify it as **Positive** or **Negative**.")

# Use text_input (works better on Colab)
user_input = st.text_input("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review before clicking the button.")
    else:
        cleaned_review = clean_text(user_input)
        vectorized_review = vectorizer.transform([cleaned_review])
        prediction = model.predict(vectorized_review)[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

st.markdown("---")
st.caption("Built using NLP, TF-IDF, Logistic Regression, and Streamlit")

