import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

print("Starting multi-model training and evaluation process...")

# --- Data Loading and Preparation ---
print("1. Loading and preparing datasets...")
try:
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')
except FileNotFoundError:
    print("Error: Dataset files not found. Please ensure True.csv and Fake.csv are in the project folder.")
    exit()

true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Text Preprocessing ---
print("2. Preprocessing text data...")
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

df['text'] = df['title'] + ' ' + df['text']
df['processed_text'] = df['text'].apply(preprocess_text)

# --- Vectorization and Splitting Data ---
print("3. Vectorizing text and splitting data...")
X = df['processed_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# --- Model Training and Comparison ---
print("4. Training multiple models and collecting metrics...")

models = {
    'Passive Aggressive Classifier': PassiveAggressiveClassifier(max_iter=50, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Multinomial Naive Bayes': MultinomialNB()
}

results = []

for name, model in models.items():
    print(f"   -> Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })
    
    # We still need to save the Passive Aggressive model for the main app's prediction page
    if name == 'Passive Aggressive Classifier':
        joblib.dump(model, 'pac_model.joblib')
        
print("Model training and evaluation for all models complete.")

# --- Saving Artifacts ---
print("5. Saving all model artifacts and comparison report...")
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(results, 'model_comparison_report.joblib')

print("Process complete. All files are saved.")
