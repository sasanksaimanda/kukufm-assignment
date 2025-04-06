
# 🎧 KukuFM Recommender System

A smart, TF-IDF-based content recommendation engine that bridges trending news with audiobooks/podcasts on **KukuFM**. This system scrapes real-time news from leading sources, extracts trending topics using NLP, and suggests relevant books/audio content accordingly. Ideal for content discovery tailored to the pulse of current events.

## 🚀 Features

- 📰 **Live News Scraping** from top national/international sources  
- 🧠 **Topic Extraction** from headlines & articles using SpaCy & NLTK  
- 📚 **TF-IDF Matching** to find books that align with trending topics  
- 🌍 Region and language-based configuration (e.g., India + English/Hindi)  
- 🧹 Automatic cleaning and preprocessing of book datasets  
- 📊 Cosine Similarity for accurate recommendations  
- 🔁 Caching for efficient news fetching  
- 🪵 Logging to track system behavior (`kukufm_recommender.log`)

## 📦 Requirements

```bash
pip install -r requirements.txt
```

Recommended packages:
- `pandas`, `numpy`, `scikit-learn`
- `nltk`, `spacy`, `beautifulsoup4`, `requests`
- SpaCy model: `en_core_web_md` or fallback to `en_core_web_sm`

Don't forget to download the necessary NLTK corpora:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## 🛠️ How to Use

```bash
python kukufm_recommender.py --book_catalog data/books.csv
```

Optional arguments:
- `--language`: Set language (default: `english`)
- `--region`: Set region (default: `india`)
- `--refresh_news`: Force refresh news scraping

## 📁 Book Catalog Format

Expected CSV columns:

- `title`
- `author`
- `description`
- `keywords`
- `genres`

Missing columns will be created with default values, but keep your data clean!

## 🧠 How It Works

1. Load and clean book data
2. Scrape top headlines & content from online news sources
3. Extract trending topics using NLP
4. Represent books using TF-IDF vectors
5. Compute cosine similarity between trending news and books
6. Recommend top-N most similar books

## 🔮 Potential Use Cases

- Personalized content suggestions on platforms like KukuFM
- Audio content discovery based on current events
- Market-aware book suggestions for digital platforms

## 📓 Future Enhancements

- Deep learning embeddings (e.g., BERT or SBERT)
- Genre-based filtering
- Sentiment-aware recommendations
- Dashboard visualization (Streamlit or Dash)

## 📜 License

MIT License. Use freely and give credit if you vibe with it. 🙌
