# IMPORTS
import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences

# PAGE CONFIG
st.set_page_config(
    page_title="BiLSTM Sentiment Analyzer",
    page_icon="🧠"
)

st.title("🧠 BiLSTM Sentiment Analyzer")

st.write(
    "Enter a review and predict sentiment."
)

#  DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# MODEL CLASS

class SentimentLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(
            embedded
        )

        hidden = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        out = self.dropout(hidden)

        out = self.fc(out)

        return out

# LOAD CONFIG
MODEL_PATH = r"E:\Customer_Feedback_Sentiment_Analyzer\saved_lstm\bilstm_sentiment_model.pth"

TOKENIZER_PATH = r"E:\Customer_Feedback_Sentiment_Analyzersaved_lstm\tokenizer.pkl"

CONFIG_PATH = r"E:\Customer_Feedback_Sentiment_Analyzer\saved_lstm\config.pkl"

# load config
with open(CONFIG_PATH, "rb") as f:
    config = pickle.load(f)

vocab_size = config["vocab_size"]
embedding_dim = config["embedding_dim"]
hidden_dim = config["hidden_dim"]
max_len = config["max_len"]

# =========================
# 6. LOAD TOKENIZER
# =========================
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# 7. LOAD MODEL
# =========================
model = SentimentLSTM(
    vocab_size,
    embedding_dim,
    hidden_dim
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

# =========================
# 8. SIMPLE PREPROCESSING
# =========================
def preprocess(text):

    text = text.lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================
# 9. PREDICTION FUNCTION
# =========================
def predict_sentiment(text):

    # preprocess
    cleaned_text = preprocess(text)

    # tokenize
    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    # padding
    padded = pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    # tensor
    input_tensor = torch.tensor(
        padded,
        dtype=torch.long
    ).to(device)

    # prediction
    with torch.no_grad():

        output = model(input_tensor)

        probability = torch.sigmoid(
            output
        ).item()

    if probability >= 0.5:

        sentiment = "Positive"

        confidence = probability * 100

    else:

        sentiment = "Negative"

        confidence = (1 - probability) * 100

    return (
        sentiment,
        confidence,
        probability,
        cleaned_text
    )

# =========================
# 10. UI
# =========================
text = st.text_area(
    "Enter Review"
)

if st.button("Predict Sentiment"):

    if text.strip() == "":

        st.warning(
            "Please enter text."
        )

    else:

        sentiment, confidence, prob, cleaned = predict_sentiment(text)

        st.write("### Processed Text")

        st.code(cleaned)

        st.write("### Prediction")

        if sentiment == "Positive":

            st.success(
                f"😊 {sentiment} Sentiment"
            )

        else:

            st.error(
                f"😞 {sentiment} Sentiment"
            )

        st.write(
            f"### Confidence: {confidence:.2f}%"
        )

        st.write(
            f"Raw Probability: {prob:.4f}"
        )