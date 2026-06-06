import string
import pandas as pd
import streamlit as st
import joblib
import re
import nltk
import numpy as np


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet


# LOAD MODEL + VECTORIZER
model = joblib.load(
    r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\LEMMA\SVC\Saved_sentiment_model_svm.pkl"
)

vectorizer = joblib.load(
    r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\LEMMA\SVC\Saved_tfidf_vectorizer.pkl"
)

config=joblib.load(r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\LEMMA\SVC\config.pkl")

ngram_range=config["ngram_range"]
max_features=config["max_features"]
min_df=config["min_df"]
max_df=config["max_df"]
C=config["C"]

contractions = {

    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",

    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",

    "won't": "will not",
    "can't": "can not",
    "couldn't": "could not",

    "shouldn't": "should not",
    "wouldn't": "would not",

    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",

    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is"
}


lemmatizer = WordNetLemmatizer()

# NEGATION HANDLING
negation_words = {"not", "no", "never"}

stop_words = set(stopwords.words("english"))


# Important sentiment words to preserve
important_words = {

        "not",
        "no",
        "nor",
        "never",

        "but",
        "very",
        "too",
        "only",

        "more",
        "most",
        "less",

        "against",
        "again",

        "down",
        "up",

        "off",
        "over",
        "under"    
}

# Remove important words from stopwords
stop_words -= important_words

# CONVERT NLTK POS TAG -> WORDNET POS TAG


def get_wordnet_pos(treebank_tag):

    
    # NOUN FAMILY

    if treebank_tag.startswith("N"):
        return wordnet.NOUN

    
    # VERB FAMILY
    
    elif treebank_tag.startswith("V"):
        return wordnet.VERB

    # ADJECTIVE FAMILY
    
    elif treebank_tag.startswith("J"):
        return wordnet.ADJ

    
    # ADVERB FAMILY
    elif treebank_tag.startswith("R"):
        return wordnet.ADV

    # DEFAULT FALLBACK
    else:
        return wordnet.NOUN

# NEGATION HANDLING
def apply_negation(tokens):
    result = []
    negate = False
    window = 3
    count = 0

    for word in tokens:
        if word in negation_words:
            negate = True
            count = window
            result.append(word)
            continue

        if negate:
            result.append("NOT_" + word)
            count -= 1
            if count == 0:
                negate = False
        else:
            result.append(word)

    return result
def preprocess(text):

    #lower
    text=text.lower()

    for short_form, full_form in contractions.items():
        text = text.replace(short_form, full_form)

    #regx
    #remove HTML tags
    text=re.sub(r"<.*?>"," ",text)

    # REMOVE URLS
    text = re.sub(r"http\S+|www\S+", " ", text)

    # KEEP ONLY LETTERS + SPACES
    text = re.sub(r"[^a-zA-Z_\s]", " ", text)
    
    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    #Tokenize
    tokens=text.split()

    #REMOVE STOPWORDS

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]
    
    # negation handling
    tokens = apply_negation(tokens)


    list_pos_tag = nltk.pos_tag(tokens)

    new_tokens=[]
    for word, pos_tag in list_pos_tag:

        # CONVERT TAG FORMAT
        wordnet_pos = get_wordnet_pos(pos_tag)

        # LEMMATIZE
        lemma = lemmatizer.lemmatize(
            word,
            pos=wordnet_pos
        )
        new_tokens.append(lemma)
    tokens=new_tokens

    return " ".join(tokens)

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
"""
test_reviews = [
    "not good",
    "not bad",
    "not very good",
    "not very bad",
    "i highly recommend you not to watch this movie",
    "i do not recommend this movie",
    "this movie is not terrible",
    "this movie is not great",
    "absolutely fantastic movie",
    "worst movie ever made",
    "good",
    "bad",
    "excellent",
    "terrible",
    "recommend",
    "not"
]
"""