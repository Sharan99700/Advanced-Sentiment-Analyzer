# advanced_sentiment_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# Load Hugging Face sentiment model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_model()

st.title("📊 Advanced Sentiment Analyzer")

st.markdown("""
This app performs **sentiment analysis** using Hugging Face Transformers.  
- Enter a single text for instant sentiment prediction.  
- Or upload a CSV file (with a column named `text`) for batch analysis.  
""")

# ---- Single Text Analysis ----
st.header("🔹 Single Text Sentiment")
text_input = st.text_area("Enter text here:")

if st.button("Analyze Text"):
    if text_input.strip():
        result = classifier(text_input)[0]
        sentiment = result['label']
        score = result['score']
        st.success(f"**Sentiment:** {sentiment} (Confidence: {score:.2f})")

        # Visualization of confidence
        labels = [sentiment, f"Not {sentiment}"]
        scores = [score, 1 - score]

        fig, ax = plt.subplots()
        ax.bar(labels, scores, color=["#4CAF50", "#FF5252"])
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please enter some text.")

# ---- Batch Analysis with CSV ----
st.header("🔹 Batch Sentiment from CSV")
uploaded_file = st.file_uploader("Upload CSV file (must have a column named 'text')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("❌ CSV must contain a column named 'text'")
    else:
        # Run sentiment analysis on each row
        st.info("Analyzing sentiments... This may take a few seconds.")
        df["Sentiment"] = df["text"].apply(lambda x: classifier(str(x))[0]['label'])
        df["Confidence"] = df["text"].apply(lambda x: classifier(str(x))[0]['score'])

        st.write("✅ Sample results:")
        st.dataframe(df.head())

        # Plot distribution
        sentiment_counts = df["Sentiment"].value_counts()
        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(sentiment_counts)
