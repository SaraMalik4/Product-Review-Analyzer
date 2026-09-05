# 🛍️ Product Review Analyzer

A Streamlit web app that lets a user search for a product by name, pulls matching reviews from the Amazon Review Polarity dataset, and uses Google's Gemini API to generate a plain-language sentiment summary — alongside the dataset's own positive/negative label breakdown for that product.

## Features

- 🔎 Search reviews by product keyword (e.g. "charger", "book", "soundtrack")
- 🧹 Text preprocessing pipeline (lowercasing, punctuation removal, stopword filtering)
- 📊 Label-based sentiment split (% positive / % negative) computed from the dataset's ground-truth polarity labels
- 🧠 AI-generated summary via Gemini 3.6 Flash covering overall sentiment, common themes, who the product suits, and a recommendation verdict
- ⚡ Cached data loading for faster repeat searches

## Tech Stack

- **Frontend/UI:** Streamlit
- **Backend logic:** Python, pandas
- **NLP preprocessing:** scikit-learn (stopword removal)
- **AI model:** Google Gemini 3.6 Flash (via the legacy `google-generativeai` SDK)
- **Data:** [Amazon Review Polarity dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) (Kaggle)

## How It Works

1. Loads a local CSV of labeled Amazon reviews (`label`, `title`, `review`)
2. Cleans and normalizes review text
3. Filters reviews containing the searched product keyword
4. Computes the % positive/negative split directly from the dataset's own labels
5. Sends the top matching reviews to Gemini with a structured prompt, asking it to summarize sentiment, themes, and give a recommendation — based on reading the review text itself, independent of the dataset labels

## Setup

```bash
git clone https://github.com/<SaraMalik4>/product-review-analyzer.git
cd product-review-analyzer
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```

Download the [Amazon Review Polarity dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) from Kaggle and place `train.csv` inside a folder named `amazon_review_polarity_csv/` in the project root.

Run the app:
```bash
streamlit run app.py
```

## Project Structure

```
product-review-analyzer/
├── app.py                # Main Streamlit app (final version)
├── experiments/          # Earlier prototypes exploring the full 22GB McAuley-Lab dataset
│   ├── load.py
│   ├── load_dataset.py
│   └── configs.py
├── key.py                # Utility script to verify .env API key loads correctly
├── requirements.txt
└── .env                  # Not committed — holds GEMINI_API_KEY
```

## Notes

- Earlier prototypes (see `experiments/`) attempted to use the full ~22GB McAuley-Lab Amazon Reviews 2023 dataset pulled live from Hugging Face. This was switched to a smaller local dataset for performance reasons.
- The dataset's own polarity labels and Gemini's generated sentiment are computed independently — one is a ground-truth statistic, the other is a language-model interpretation of the review text — and are shown side by side rather than merged.

## Known Limitations

- **Searching specific modern products (e.g. "iPhone 12", "AirPods") returns no results.** The Amazon Review Polarity dataset was compiled c. 2013–2015 from a broad, generic mix of product categories (books, soundtracks, chargers, TVs, appliances). It predates products like the iPhone 12 entirely, so no keyword match is possible — this is a dataset ceiling, not a bug in the search logic. Generic category terms (e.g. "charger", "book", "soundtrack") work as expected.
- Search originally checked only the `review` text column. Product names often appear in the `title` field instead (e.g. a TV model number in the title, generic wording in the review body), so title text is now included in the searchable field alongside the review body.

## Screenshots

*(to add)*

## Future Improvements

- Swap in a dataset with structured product metadata (e.g. McAuley-Lab's Amazon Reviews 2023) to support real, specific product name search — this was attempted early on (see `experiments/`) but dropped due to the dataset's ~22GB size and local performance constraints
- Compare Gemini's inferred sentiment against the dataset's ground-truth labels to evaluate agreement
- Support fuzzy/partial product name matching
- Deploy publicly (Streamlit Community Cloud)
