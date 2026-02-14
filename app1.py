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
                "image": img if img else "https://via.placeholder.com/150"
            })
    df = pd.DataFrame(articles)
    df["summary"] = df["summary"].apply(clean_html)
    df["clean_text"] = (df["title"].fillna('') + " " + df["summary"].fillna('')).apply(preprocess_text)
    df["sentiment"] = df["clean_text"].apply(get_sentiment)
    df["category"] = df["clean_text"].apply(classify_article)
    return df

# =====================================================
# Session State
# =====================================================
if "selected_category" not in st.session_state:
    st.session_state.selected_category = "Politics"
if "page" not in st.session_state:
    st.session_state.page = 0

# =====================================================
# Dark/Light Mode Toggle
# =====================================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"
theme = st.radio("Theme:", ["Light", "Dark"], index=0)
st.session_state.theme = theme
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
.scrolling-wrapper {{
  display: flex;
  overflow-x: auto;
  padding: 10px;
}}
.card-horizontal {{
  flex: 0 0 auto;
  width: 250px;
  margin-right: 10px;
  border-radius: 10px;
  padding: 10px;
  border: 1px solid #ccc;
  transition: transform 0.2s, box-shadow 0.2s;
}}
.card-horizontal:hover {{
  transform: scale(1.05);
  box-shadow: 0 8px 16px rgba(0,0,0,0.3);
}}
.card-horizontal img {{
  width: 100%;
  border-radius: 8px;
}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Top Nav Bar
# =====================================================
st.title("🇰🇪 Kenyan AI News Aggregator")
categories = ["Politics","Sports","Business","Technology","Health","Other"]
cols = st.columns(len(categories))
for i,cat in enumerate(categories):
    if cols[i].button(cat):
        st.session_state.selected_category = cat
        st.session_state.page = 0

# =====================================================
# Fetch news once
# =====================================================
news_df = scrape_rss(rss_feeds)

# ML Classification
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
# Filter and paginate
# =====================================================
articles_per_page = 6
filtered_df = news_df[news_df["ml_category"]==st.session_state.selected_category]
total_pages = (len(filtered_df)-1)//articles_per_page +1
start_idx = st.session_state.page*articles_per_page
end_idx = start_idx+articles_per_page
page_articles = filtered_df.iloc[start_idx:end_idx]

# =====================================================
# Display cards (3 columns)
# =====================================================
n_cols = 3
for i in range(0,len(page_articles),n_cols):
    cols = st.columns(n_cols)
    for j,(_,row) in enumerate(page_articles.iloc[i:i+n_cols].iterrows()):
        with cols[j]:
            st.markdown(f"""
            <div class='card'>
                <img src='{row['image']}' width='100%'>
                <h4><a href='{row['link']}' target='_blank'>{row['title']}</a></h4>
                <p>{row['source']} | <span style='color:{sentiment_color(row['sentiment'])}'>{row['sentiment']}</span></p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# Pagination Buttons
# =====================================================
col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("⬅️ Previous") and st.session_state.page>0:
        st.session_state.page-=1
        st.experimental_rerun()
with col_next:
    if st.button("Next ➡️") and st.session_state.page<total_pages-1:
        st.session_state.page+=1
        st.experimental_rerun()

# =====================================================
# Recommendations Carousel
# =====================================================
st.subheader("Recommended Articles for this Category")
tfidf_matrix = tfidf_vectorizer.transform(filtered_df["clean_text"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

for idx, (_, article) in enumerate(page_articles.iterrows()):
    st.markdown(f"**Recommendations for:** {article['title']}")
    sim_scores = list(enumerate(cosine_sim[filtered_df.index.get_loc(article.name)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    rec_articles = filtered_df.iloc[[i[0] for i in sim_scores]]
    scroll_html = "<div class='scrolling-wrapper'>"
    for _, rec in rec_articles.iterrows():
        scroll_html += f"""
        <div class='card-horizontal'>
            <img src='{rec['image']}'>
            <h5><a href='{rec['link']}' target='_blank'>{rec['title']}</a></h5>
            <p>{rec['ml_category']} | <span style='color:{sentiment_color(rec['sentiment'])}'>{rec['sentiment']}</span></p>
        </div>
        """
    scroll_html += "</div>"
    st.markdown(scroll_html, unsafe_allow_html=True)
