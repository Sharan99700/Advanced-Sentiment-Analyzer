# advanced_sentiment_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# Load Hugging Face sentiment model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_model()

st.title("📊 Advanced Sentiment Analyzer (with Interactive Graphs)")

st.markdown("""
This app performs **sentiment analysis** using Hugging Face Transformers.  
- Enter a single text for instant prediction with visualization.  
- Or upload a CSV file (with a column named `text`) for batch analysis with charts.  
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

        # Interactive donut chart for confidence
        fig = px.pie(
            values=[score, 1 - score],
            names=[sentiment, f"Not {sentiment}"],
            hole=0.5,
            title="Confidence Distribution",
            color=[sentiment, f"Not {sentiment}"],
            color_discrete_sequence=["#4CAF50", "#FF5252"]
        )
        st.plotly_chart(fig, use_container_width=True)

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
        st.info("Analyzing sentiments... Please wait.")
        df["Sentiment"] = df["text"].apply(lambda x: classifier(str(x))[0]['label'])
        df["Confidence"] = df["text"].apply(lambda x: classifier(str(x))[0]['score'])

        st.write("✅ Sample results:")
        st.dataframe(df.head())

        # Pie chart for sentiment distribution
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        pie_fig = px.pie(
            sentiment_counts,
            values="Count",
            names="Sentiment",
            hole=0.4,
            title="Sentiment Distribution (Pie Chart)"
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        # Interactive bar chart
        bar_fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            text="Count",
            color="Sentiment",
            title="Sentiment Distribution (Bar Chart)"
        )
        bar_fig.update_traces(textposition="outside")
        st.plotly_chart(bar_fig, use_container_width=True)
