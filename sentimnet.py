# advanced_sentiment_app.py
import streamlit as st
from transformers import pipeline

# Load sentiment analysis pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

st.title("Advanced Sentiment Analyzer")

text_input = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        results = classifier(text_input)
        sentiment = results[0]['label']
        score = results[0]['score']
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {score:.2f}")
