import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import streamlit as st
from gensim.models import Word2Vec

# Ensure the necessary NLTK data is available
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Define the text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return " ".join(lemmatized_words)

# Define the function to create an averaged word vector for a comment
def comment_vector(cleaned_comment, model):
    words = cleaned_comment.split()
    vector = np.mean(
        [model.wv[word] for word in words if word in model.wv]
        or [np.zeros(model.vector_size)],
        axis=0,
    )
    return vector

# Function to preprocess and transform a comment for prediction
def preprocess_and_predict(comment, model):
    cleaned_comment = clean_text(comment)
    comment_tfidf = tfidf_vectorizer.transform([cleaned_comment])
    comment_w2v = comment_vector(cleaned_comment, model_w2v).reshape(1, -1)
    combined_features = np.concatenate((comment_tfidf.toarray(), comment_w2v), axis=1)
    prediction = model.predict(combined_features)
    return prediction[0]

# Load models and vectorizers from relative paths
log_reg_model = joblib.load("logistic_regression.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
model_w2v = Word2Vec.load("word2vec_model")

# Streamlit interface
st.title("Comment Toxicity Prediction")

# Input comment
user_comment = st.text_area("Enter your comment:")

if st.button("Predict"):
    if user_comment:
        prediction = preprocess_and_predict(user_comment, log_reg_model)
        result = "toxic" if prediction == 1 else "non-toxic"
        st.write(f"The comment is predicted to be: {result}")
    else:
        st.write("Please enter a comment.")
