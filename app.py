import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


try:
    img_uri = get_base64_of_bin_file("assets/background.jpg")
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{img_uri}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;
        }}
        .stTextArea, .stButton > button {{
            background-color: rgba(38, 39, 48, 0.7);
            border-radius: 10px;
        }}
        body, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stSuccess, .stError, .stWarning {{
            color: #FAFAFA;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning("Background image not found. Please ensure 'background.jpg' is in the 'assets' folder.")
   
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Helper function for text preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(filtered_tokens)

# --- Load the saved model and vectorizer ---
try:
    pac_model = joblib.load('pac_model.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    st.error("Error: Model files ('pac_model.joblib' or 'tfidf_vectorizer.joblib') not found.")
    st.error("Please run 'train_model.py' first to train and save the model.")
    st.stop()

# --- Streamlit UI ---
st.title('Fake News Detection System')
st.markdown("---")

st.markdown("### Enter a News Article Below")
st.write("Paste the full content of a news article into the text box below. "
         "The system will then predict whether it's likely real or fake news.")

user_input = st.text_area(
    "News Article Content",
    height=300,
    placeholder="e.g., 'WASHINGTON (Reuters) - President Donald Trump today signed a bill...'"
)

if st.button('Predict'):
    if user_input:
        processed_input = preprocess_text(user_input)
        vectorized_input = tfidf_vectorizer.transform([processed_input])
        prediction = pac_model.predict(vectorized_input)
        
        st.markdown("### Prediction Result:")
        if prediction[0] == 1:
            st.success("This article is likely **Real News** ✅")
            st.balloons()
        else:
            st.error("This article is likely **Fake News** 🚫")
            st.snow()
            
    else:
        st.warning("Please enter some text to get a prediction.")

st.markdown("---")
st.markdown("Project by: SYED TASNEEM KOUSAR")
st.markdown("Technologies: Python, Pandas, NLTK, Scikit-learn, Streamlit")
