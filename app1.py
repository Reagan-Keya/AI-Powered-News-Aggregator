!pip install feedparser beautifulsoup4 pandas scikit-learn nltk vaderSentiment streamlit --quiet

import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')

# =====================================================
# PHASE 2: IMPORT LIBRARIES
# =====================================================
import feedparser
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import streamlit as st
from datetime import datetime

# =====================================================
# PHASE 3: SCRAPE KENYAN RSS FEEDS
# =====================================================
rss_feeds = {
    "Daily Nation": "https://nation.africa/kenya/rss",
    "The Standard": "https://www.standardmedia.co.ke/rss/headlines.php",
    "The Star": "https://www.the-star.co.ke/rss/",
    "Capital FM": "https://www.capitalfm.co.ke/news/feed/"
}

def scrape_rss(feeds):
    articles = []
    for source, url in feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            article = {
                "source": source,
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", "")
            }
            articles.append(article)
    return pd.DataFrame(articles)

# =====================================================
# PHASE 4: TEXT CLEANING & PREPROCESSING
# =====================================================
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

def clean_html(text):
    return BeautifulSoup(str(text), "html.parser").get_text()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# =====================================================
# PHASE 5: KEYWORD-BASED CATEGORY (BASELINE)
# =====================================================
def classify_article(text):
    if any(word in text for word in ["election", "president", "parliament", "governor", "politics"]):
        return "Politics"
    elif any(word in text for word in ["football", "sports", "match", "league", "goal"]):
        return "Sports"
    elif any(word in text for word in ["market", "economy", "business", "finance", "bank"]):
        return "Business"
    elif any(word in text for word in ["technology", "tech", "ai", "digital", "software"]):
        return "Technology"
    elif any(word in text for word in ["health", "hospital", "covid", "disease", "medical"]):
        return "Health"
    else:
        return "Other"

# =====================================================
# PHASE 6: STREAMLIT APP
# =====================================================
st.title("🇰🇪 Kenyan AI News Aggregator")
st.write("AI-powered news aggregator with ML classification, sentiment analysis, and recommendations.")

if st.button("Scrape Latest News"):
    with st.spinner("Fetching news..."):
        # Scrape
        news_df = scrape_rss(rss_feeds)
        news_df["summary"] = news_df["summary"].apply(clean_html)
        news_df["clean_text"] = (news_df["title"].fillna('') + " " + news_df["summary"].fillna('')).apply(preprocess_text)
        news_df["sentiment"] = news_df["clean_text"].apply(get_sentiment)
        news_df["category"] = news_df["clean_text"].apply(classify_article)
        
        # =====================================================
        # TF-IDF & ML CLASSIFICATION
        # =====================================================
        # Remove 'Other' for ML
        ml_df = news_df[news_df["category"] != "Other"].copy()
        X = ml_df["clean_text"]
        y = ml_df["category"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train_tfidf, y_train)

        y_pred = lr_model.predict(X_test_tfidf)
        st.write("**ML Classification Accuracy:**", accuracy_score(y_test, y_pred))
        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        # Predict full dataset
        news_df["ml_category"] = lr_model.predict(tfidf_vectorizer.transform(news_df["clean_text"]))

        # =====================================================
        # Save model and vectorizer for deployment
        # =====================================================
        with open("logistic_news_model.pkl", "wb") as f:
            pickle.dump(lr_model, f)
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf_vectorizer, f)

        st.success(f"Scraped {len(news_df)} articles successfully!")

        # Show top 10 news
        st.subheader("Top 10 Latest News")
        st.dataframe(news_df[["title", "source", "ml_category", "sentiment", "link"]].head(10))

        # =====================================================
        # 7️⃣ Recommendations
        # =====================================================
        st.subheader("Get Article Recommendations")
        article_title = st.selectbox("Select an article:", news_df["title"])
        if article_title:
            idx = news_df[news_df["title"] == article_title].index[0]
            
            # Recommendation engine
            tfidf_matrix = tfidf_vectorizer.transform(news_df["clean_text"])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            similarity_scores = list(enumerate(cosine_sim[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
            recommended_indices = [i[0] for i in similarity_scores]
            recommendations = news_df.iloc[recommended_indices][["title", "source", "ml_category", "sentiment", "link"]]

            for i, row in recommendations.iterrows():
                st.markdown(f"**{row['title']}** ({row['ml_category']}, {row['sentiment']}) - [{row['source']}]({row['link']})")

# =====================================================
# END OF STREAMLIT APP
# =====================================================
