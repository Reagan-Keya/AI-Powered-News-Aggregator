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
from streamlit.components.v1 import html

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
    if any(word in text for word in ["election","president","parliament","governor","politics"]):
        return "Politics"
    elif any(word in text for word in ["football","sports","match","league","goal"]):
        return "Sports"
    elif any(word in text for word in ["market","economy","business","finance","bank"]):
        return "Business"
    elif any(word in text for word in ["technology","tech","ai","digital","software"]):
        return "Technology"
    elif any(word in text for word in ["health","hospital","covid","disease","medical"]):
        return "Health"
    else:
        return "Other"

def sentiment_color(sent):
    if sent=="Positive": return "green"
    if sent=="Negative": return "red"
    return "orange"

# =====================================================
# Scrape RSS
# =====================================================
@st.cache_data(ttl=600)
def scrape_rss(feeds):
    articles = []
    for source,url in feeds.items():
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
                "image": img if img else "https://via.placeholder.com/300x180"
            })
    df = pd.DataFrame(articles)
    df["summary"] = df["summary"].apply(clean_html)
    df["clean_text"] = (df["title"].fillna('') + " " + df["summary"].fillna('')).apply(preprocess_text)
    df["sentiment"] = df["clean_text"].apply(get_sentiment)
    df["category"] = df["clean_text"].apply(classify_article)
    return df

news_df = scrape_rss(rss_feeds)

# =====================================================
# ML Classification
# =====================================================
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

# =====================================================
# Session State
# =====================================================
if "selected_category" not in st.session_state:
    st.session_state.selected_category = "Politics"

# =====================================================
# Top Nav Bar
# =====================================================
st.title("🇰🇪 Kenyan AI News - Netflix Style")
categories = ["Politics","Sports","Business","Technology","Health","Other"]
cols = st.columns(len(categories))
for i,cat in enumerate(categories):
    if cols[i].button(cat):
        st.session_state.selected_category = cat

# =====================================================
# Generate Carousel HTML
# =====================================================
def generate_carousel_html(df_category, title):
    tfidf_matrix = tfidf_vectorizer.transform(df_category["clean_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    html_content = f"<h3 style='color:white;margin-left:10px;'>{title}</h3>"
    html_content += "<div style='display:flex; overflow-x:auto; padding:10px;'>"
    
    for idx, (_, row) in enumerate(df_category.iterrows()):
        # Recommendations for hover
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
        rec_titles = [df_category.iloc[i[0]]["title"] for i in sim_scores]
        rec_text = "<br>".join(rec_titles)
        
        html_content += f"""
        <div style='flex:0 0 auto; margin-right:10px; position:relative; width:300px; cursor:pointer; transition:0.3s;'>
            <img src='{row['image']}' style='width:100%; height:180px; object-fit:cover; border-radius:10px;'>
            <div style='position:absolute; bottom:0; width:100%; padding:10px; background:linear-gradient(to top, rgba(0,0,0,0.7), transparent); color:white; font-weight:bold;' title='{rec_text}'>
                {row['title']}
            </div>
        </div>
        """
    html_content += "</div>"
    return html_content

# =====================================================
# Render Netflix-style Carousels
# =====================================================
st.markdown("<style>body{background-color:#111;color:white;}</style>", unsafe_allow_html=True)
for cat in categories:
    df_cat = news_df[news_df["ml_category"]==cat]
    if len(df_cat) > 0:
        carousel_html = generate_carousel_html(df_cat, cat)
        html(carousel_html, height=250)
