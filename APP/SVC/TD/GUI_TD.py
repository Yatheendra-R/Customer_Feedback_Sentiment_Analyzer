import string
import pandas as pd
import streamlit as st
import joblib
import re
import nltk
import numpy as np


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

model = joblib.load(
    r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\Tonedownlemma\SVC\Saved_sentiment_model_svm.pkl"
)

vectorizer = joblib.load(
    r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\Tonedownlemma\SVC\Saved_tfidf_vectorizer.pkl"
)

config=joblib.load(r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\Tonedownlemma\SVC\config.pkl")

ngram_range=config["ngram_range"]
max_features=config["max_features"]
min_df=config["min_df"]
max_df=config["max_df"]
C=config["C"] 


# SIMPLE CONTRACTIONS
contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is"
}

# CLEANING FUNCTION 
def preprocess(text):

    # 1. lowercase
    text = text.lower()

    # 2. expand contractions
    for k, v in contractions.items():
        text = text.replace(k, v)

    # 3. remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # 4. remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 5. keep letters + apostrophes (important for "don't")
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # 6. normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# STREAMLIT UI
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠"
)

st.title("🧠 Customer Feedback Sentiment Analyzer")

st.write(
    "Enter a review and predict sentiment."
)

text = st.text_area(
    "Enter Review"
)

if st.button("Predict Sentiment"):

    if text.strip() == "":

        st.warning("Please enter text.")

    else:

        # preprocess
        cleaned_text = preprocess(text)

        st.write("### Processed Text")
        st.code(cleaned_text)

        # vectorize
        vec = vectorizer.transform([cleaned_text])

        # prediction
        pred = model.predict(vec)[0]

        # confidence score
        score = model.decision_function(vec)[0]

        confidence = 1 / (1 + np.exp(-abs(score)))

        confidence_percent = confidence * 100

        # output
        st.write("### Prediction")

        if pred == "positive":

            st.success(
                f"😊 Positive Sentiment"
            )

        else:

            st.error(
                f"😞 Negative Sentiment"
            )

        st.write(
            f"### Confidence: {confidence_percent:.2f}%"
        )
