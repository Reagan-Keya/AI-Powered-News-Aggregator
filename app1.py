import streamlit as st
import pandas as pd
import feedparser
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import nltk

# =====================================================
# NLTK Setup
# =====================================================
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
download_nltk_data()

# =====================================================
# Page Config
# =====================================================
st.set_page_config(page_title="Kenyan AI News", layout="wide")

# =====================================================
# Dark/Light Mode Toggle
# =====================================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"
theme = st.radio("Select Theme:", ["Light", "Dark"], index=0)
st.session_state.theme = theme

# Inject custom CSS for cards and hover effects
if st.session_state.theme == "dark":
    bg_color = "#0e1117"
    text_color = "#f5f5f5"
    card_bg = "#1f2937"
else:
    bg_color = "#ffffff"
    text_color = "#000000"
    card_bg = "#f8f8f8"

st.markdown(f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .card {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    .card:hover {{
        transform: scale(1.03);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }}
    a {{
        text-decoration: none;
        color: inherit;
    }}
    </style>
""", unsafe_allow_html=True)

# =====================================================
# Globals
# =====================================================
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

rss_feeds = {
    "Daily Nation": "https://nation.africa/kenya/rss",
    "The Standard": "https://www.standardmedia.co.ke/rss/headlines.php",
    "The Star": "https://www.the-star.co.ke/rss/",
    "Capital FM": "https://www.capitalfm.co.ke/news/feed/"
}

# =====================================================
# Functions
# =====================================================
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

def sentiment_color(sent):
    if sent == "Positive": return "green"
    if sent == "Negative": return "red"
    return "orange"

def scrape_rss(feeds):
    articles = []
    for source, url in feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            img = None
            if 'media_content' in entry:
                img = entry.media_content[0]['url']
            elif 'media_thumbnail' in entry:
                img = entry.media_thumbnail[0]['url']
            articles.append({
                "source": source,
                "title": entry.get("title",""),
                "summary": entry.get("summary",""),
                "link": entry.get("link",""),
                "image": img if img else "https://via.placeholder.com/150"
            })
    return pd.DataFrame(articles)

# =====================================================
# Session State Init
# =====================================================
if "selected_category" not in st.session_state:
    st.session_state.selected_category = "Politics"
if "page" not in st.session_state:
    st.session_state.page = 0

# =====================================================
# Top Navigation
# =====================================================
st.title("🇰🇪 Kenyan AI News Aggregator")
categories = ["Politics","Sports","Business","Technology","Health","Other"]
category_icons = {
    "Politics":"https://cdn-icons-png.flaticon.com/128/1055/1055646.png",
    "Sports":"https://cdn-icons-png.flaticon.com/128/3135/3135715.png",
    "Business":"https://cdn-icons-png.flaticon.com/128/3135/3135713.png",
    "Technology":"https://cdn-icons-png.flaticon.com/128/1055/1055651.png",
    "Health":"https://cdn-icons-png.flaticon.com/128/2965/2965567.png",
    "Other":"https://cdn-icons-png.flaticon.com/128/2910/2910765.png"
}

cols = st.columns(len(categories))
for i, cat in enumerate(categories):
    if cols[i].button(f"{cat}"):
        st.session_state.selected_category = cat
        st.session_state.page = 0

# =====================================================
# Scrape News Button
# =====================================================
if st.button("Scrape Latest News"):
    with st.spinner("Fetching news..."):
        news_df = scrape_rss(rss_feeds)
        news_df["summary"] = news_df["summary"].apply(clean_html)
        news_df["clean_text"] = (news_df["title"].fillna('') + " " + news_df["summary"].fillna('')).apply(preprocess_text)
        news_df["sentiment"] = news_df["clean_text"].apply(get_sentiment)
        news_df["category"] = news_df["clean_text"].apply(classify_article)

        # ML
        @st.cache_resource
        def train_ml_model(df):
            ml_df = df[df["category"]!="Other"].copy()
            X = ml_df["clean_text"]
            y = ml_df["category"]
            tfidf_vectorizer = TfidfVectorizer(max_features=5000)
            X_tfidf = tfidf_vectorizer.fit_transform(X)
            lr_model = LogisticRegression(max_iter=1000)
            lr_model.fit(X_tfidf, y)
            return lr_model, tfidf_vectorizer

        lr_model, tfidf_vectorizer = train_ml_model(news_df)
        news_df["ml_category"] = lr_model.predict(tfidf_vectorizer.transform(news_df["clean_text"]))

        st.success(f"Scraped {len(news_df)} articles successfully!")

        # Pagination
        articles_per_page = 6
        filtered_df = news_df[news_df["ml_category"]==st.session_state.selected_category]
        total_pages = (len(filtered_df)-1)//articles_per_page +1
        start_idx = st.session_state.page*articles_per_page
        end_idx = start_idx+articles_per_page
        page_articles = filtered_df.iloc[start_idx:end_idx]

        # Display cards
        n_cols = 3
        for i in range(0, len(page_articles), n_cols):
            cols = st.columns(n_cols)
            for j, (_, row) in enumerate(page_articles.iloc[i:i+n_cols].iterrows()):
                with cols[j]:
                    st.markdown(f"""
                    <div class='card'>
                        <img src='{row['image']}' width='100%'>
                        <h4><a href='{row['link']}' target='_blank'>{row['title']}</a></h4>
                        <p>{row['source']} | <span style='color:{sentiment_color(row['sentiment'])}'>{row['sentiment']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)

        # Pagination buttons
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Previous") and st.session_state.page>0:
                st.session_state.page-=1
                st.experimental_rerun()
        with col_next:
            if st.button("Next ➡️") and st.session_state.page<total_pages-1:
                st.session_state.page+=1
                st.experimental_rerun()

        # Recommendations
        st.subheader("Recommended Articles")
        article_title = st.selectbox("Select an article:", news_df["title"])
        if article_title:
            idx = news_df[news_df["title"]==article_title].index[0]
            tfidf_matrix = tfidf_vectorizer.transform(news_df["clean_text"])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            similar_idx = sorted(list(enumerate(cosine_sim[idx])), key=lambda x:x[1], reverse=True)[1:6]
            recommended = news_df.iloc[[i[0] for i in similar_idx]][["title","source","ml_category","sentiment","link","image"]]

            rec_cols = st.columns(len(recommended))
            for k, (_, row) in enumerate(recommended.iterrows()):
                with rec_cols[k]:
                    st.markdown(f"""
                    <div class='card'>
                        <img src='{row['image']}' width='100%'>
                        <h5><a href='{row['link']}' target='_blank'>{row['title']}</a></h5>
                        <p>{row['ml_category']} | <span style='color:{sentiment_color(row['sentiment'])}'>{row['sentiment']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
