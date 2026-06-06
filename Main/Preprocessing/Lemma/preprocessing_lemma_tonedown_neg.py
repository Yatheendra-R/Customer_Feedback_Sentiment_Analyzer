import re
import pandas as pd

# =========================
# CONTRACTIONS
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
    "mustn't": "must not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are"
}

# =========================
# NEGATION WORDS
# =========================
NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "cannot"
}

NEGATION_WINDOW = 4

# =========================
# PREPROCESSING FUNCTION
# =========================
def preprocess_lstm(text):

    # lowercase
    text = str(text).lower()

    # expand contractions
    for k, v in contractions.items():
        text = text.replace(k, v)

    # remove html
    text = re.sub(r"<.*?>", " ", text)

    # remove urls
    text = re.sub(r"http\S+|www\S+", " ", text)

    # keep only letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # =========================
    # NEGATION PROPAGATION
    # =========================
    words = text.split()

    result = []

    negate_count = 0

    for word in words:

        if word in NEGATION_WORDS:

            result.append(word)

            negate_count = NEGATION_WINDOW

            continue

        if negate_count > 0:

            result.append("NOT_" + word)

            negate_count -= 1

        else:

            result.append(word)

    return " ".join(result)

# =========================
# LOAD DATA
# =========================
df=pd.read_csv(r"E:\Customer_Feedback_Sentiment_Analyzer\Data\IMDB_dataset.csv")


print("Before Cleaning:\n")
print(df["review"][0])

# =========================
# APPLY PREPROCESSING
# =========================
df["review"] = df["review"].apply(
    preprocess_lstm
)

print("\nAfter Cleaning:\n")
print(df["review"][0])

# =========================
# SAVE CLEANED DATA
# =========================
output_path = (
    r"E:\Customer_Feedback_Sentiment_Analyzer\Data\cleaned_imdb_negation.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nDataset saved successfully!")
print(output_path)

# =========================
# QUICK TESTS
# =========================
tests = [
    "not good",
    "not bad",
    "not very good",
    "not very bad",
    "i do not recommend this movie",
    "i highly recommend you not to watch this movie",
    "this movie is not terrible",
    "this movie is not great"
]

print("\n========================")
print("NEGATION TESTS")
print("========================")

for t in tests:

    print("\nINPUT : ", t)
    print("OUTPUT: ", preprocess_lstm(t))