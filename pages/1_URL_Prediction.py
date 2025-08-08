import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
import warnings
import base64

warnings.filterwarnings("ignore")

st.title("News Article URL Predictor")
st.markdown("---")
st.write("Paste the URL of a news article below. The app will scrape the content and predict if it's real or fake news.")

# --- Helper function for text preprocessing (must be same as in app.py and train_model.py) ---
@st.cache_data
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(filtered_tokens)

# --- Load Model and Vectorizer (cached for speed) ---
@st.cache_data
def load_model_and_vectorizer():
    try:
        pac_model = joblib.load('pac_model.joblib')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
        return pac_model, tfidf_vectorizer
    except FileNotFoundError:
        st.error("Error: Model files not found. Please run 'train_model.py' first.")
        st.stop()

pac_model, tfidf_vectorizer = load_model_and_vectorizer()

# --- Helper function for web scraping ---
@st.cache_data
def scrape_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Simple method to find and concatenate paragraph text
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        
        # Check if text was successfully scraped
        if not article_text.strip():
            st.warning("Could not find any readable text. The page may be a video, image, or require user interaction.")
            return None
        
        return article_text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        st.error(f"Error scraping content: {e}")
        return None

# --- Streamlit UI for URL prediction ---
url_input = st.text_input(
    "Enter a news article URL:",
    placeholder="e.g., https://www.reuters.com/article/example"
)

if st.button('Predict from URL'):
    if url_input:
        with st.spinner('Scraping content and making prediction...'):
            article_text = scrape_article_text(url_input)
            
            if article_text:
                processed_input = preprocess_text(article_text)
                vectorized_input = tfidf_vectorizer.transform([processed_input])
                prediction = pac_model.predict(vectorized_input)
                
                st.markdown("### Prediction Result:")
                if prediction[0] == 1:
                    st.success("This article is likely **Real News** ✅")
                    st.balloons() # Added balloons for correct prediction
                else:
                    st.error("This article is likely **Fake News** 🚫")
                    st.snow() # Added snow for incorrect prediction
    else:
        st.warning("Please enter a URL to get a prediction.")
