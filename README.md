📰 Fake News Detector

A machine learning project that classifies news articles as Fake or Real. This project leverages Natural Language Processing (NLP) techniques to preprocess text data and train machine learning models to improve the reliability of news consumption.

🚀 Project Overview

The goal of this project is to automatically identify misleading or fake news using data-driven techniques. The system works by:

Collecting and cleaning textual data.

Extracting features using NLP methods (TF-IDF, Bag of Words).

Training ML models such as Logistic Regression, Naive Bayes, and XGBoost.

Evaluating model performance and selecting the best classifier.

⚙️ Tech Stack

Programming Language: Python 🐍

Libraries:

Pandas, NumPy → Data handling

NLTK, Scikit-learn → NLP & ML models

XGBoost → Boosted tree classification

Flask / Streamlit → Web app deployment

📊 Dataset

The project uses datasets such as:

Kaggle’s Fake News Dataset (contains real & fake news labeled data).

Preprocessed with tokenization, stopword removal, lemmatization.

Columns typically include:

id → Unique identifier

title → Headline of the article

text → Main content of the article

label → (0 = Real, 1 = Fake)

🔎 Model Workflow

Preprocessing

Lowercasing, punctuation removal, stopword removal.

Tokenization & lemmatization.

TF-IDF feature extraction.

Training Models

Logistic Regression

Naive Bayes

XGBoost

Evaluation Metrics

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Deployment

Web app interface to input news & get predictions in real-time.

💻 How to Run the Project

Clone this repo:

git clone https://github.com/yourusername/Fake-News-Detector.git
cd Fake-News-Detector


Install dependencies:

pip install -r requirements.txt


Run Jupyter notebook for training:

jupyter notebook notebooks/fake_news_detection.ipynb


Launch the app (Flask / Streamlit):

python app.py

🏆 Results

Logistic Regression → ~92% accuracy

Naive Bayes → ~90% accuracy

XGBoost → ~95% accuracy (best model)

👥 Contributors

Tasneem Kousar



📜 License

This project is licensed under the MIT License.
