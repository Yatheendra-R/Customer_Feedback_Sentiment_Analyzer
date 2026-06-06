import re
import pandas as pd

# =========================
# SIMPLE CONTRACTIONS
# =========================
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

# =========================
# CLEANING FUNCTION (BEST FOR LSTM)
# =========================
def preprocess_lstm(text):

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


# =========================
# LOAD DATA
# =========================
df=pd.read_csv(r"E:\Customer_Feedback_Sentiment_Analyzer\Data\IMDB_dataset.csv")

print("Before Cleaning Sample:\n", df["review"][0])

df["review"] = df["review"].apply(preprocess_lstm)

print("\nAfter Cleaning Sample:\n", df["review"][0])


# =========================
# SAVE CLEANED DATA
# =========================
output_path = r"E:\Customer_Feedback_Sentiment_Analyzer\Data\cleaned_imdb_lstm_td.csv"
df.to_csv(output_path, index=False)

print("\nLSTM Cleaned dataset saved successfully!")