import streamlit as st
import pandas as pd
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- IMPROVEMENT: fail gracefully if the key is missing ---
if not api_key:
    st.error("❌ GEMINI_API_KEY not found. Please check your .env file.")
    st.stop()

# Set up Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3.6-flash")

# --- FIX: csv_path needs to be defined before load_data() uses it ---
csv_path = "amazon_review_polarity_csv/train.csv"

# Load dataset from local CSV
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, header=None, engine='python', nrows=10000)
    df.columns = ["label", "title", "review"]
    df["sentiment"] = df["label"].map({1: "negative", 2: "positive"})
    return df

df = load_data(csv_path)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

# Streamlit UI
st.title("🛍️ Product Review Analyzer (Gemini-powered)")
st.markdown("Enter a product name to analyze real Amazon reviews and get an AI summary.")

product_name = st.text_input("🔎 Enter product name (e.g., iphone 12, airpods, charger)")
analyze_button = st.button("Analyze Reviews")

# AI summarizer
def summarize_reviews(product_name, reviews_df):
    combined_reviews = "\n".join(reviews_df["cleaned_review"].tolist())

    prompt = f"""
You are an AI assistant summarizing Amazon customer reviews.

Product: {product_name}

Here are some actual customer reviews:
{combined_reviews}

Please summarize the reviews by mentioning:
- Overall sentiment (positive/negative/mixed)
- Common themes (likes, dislikes)
- Who would benefit from this product (e.g., students, travelers)
- Whether this product is worth recommending

Respond in 5–7 sentences.
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Main logic
if analyze_button and product_name:
    with st.spinner("🧠 Analyzing reviews..."):
        df["cleaned_review"] = df["review"].astype(str).apply(preprocess_text)
        keyword = preprocess_text(product_name)
        filtered = df[df["cleaned_review"].str.contains(keyword)]

        if filtered.empty:
            st.warning("😕 No matching reviews found. Try a different product name.")
        else:
            st.success(f"✅ Found {len(filtered)} matching reviews.")
            top_reviews = filtered.head(30)

            # --- IMPROVEMENT: surface the dataset's own polarity labels ---
            sentiment_counts = filtered["sentiment"].value_counts(normalize=True) * 100
            pos_pct = sentiment_counts.get("positive", 0)
            neg_pct = sentiment_counts.get("negative", 0)

            col1, col2 = st.columns(2)
            col1.metric("👍 Labeled Positive", f"{pos_pct:.1f}%")
            col2.metric("👎 Labeled Negative", f"{neg_pct:.1f}%")

            summary = summarize_reviews(product_name, top_reviews)
            st.markdown("### 📝 AI Summary")
            st.write(summary)