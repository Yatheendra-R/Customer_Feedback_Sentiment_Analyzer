import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences

# PAGE CONFIG

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 Movie Review Sentiment Analysis")
st.write("Enter a review and predict sentiment using Attention BiLSTM")

# PREPROCESSING

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

def preprocess_lstm(text):

    text = text.lower()

    for k, v in contractions.items():
        text = text.replace(k, v)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

#model class
class AttentionBiLSTM(nn.Module):

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

        self.embedding_dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.attention = nn.Linear(
            hidden_dim * 2,
            1
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            hidden_dim * 2,
            1
        )

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):

        mask = (x != 0)

        embedded = self.embedding(x)

        embedded = self.embedding_dropout(
            embedded
        )

        lstm_out, (hidden, cell) = self.lstm(
            embedded
        )

        scores = self.attention(
            lstm_out
        ).squeeze(-1)

        scores = scores.masked_fill(
            ~mask,
            -1e9
        )

        attn_weights = torch.softmax(
            scores,
            dim=1
        )

        context_vector = torch.sum(
            lstm_out *
            attn_weights.unsqueeze(-1),
            dim=1
        )

        out = self.dropout(
            context_vector
        )

        out = self.fc(out)

        return out

# LOAD FILES
MODEL_PATH = r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\Tonedownlemma\GloVe_stack_BiLSTM_Attention\attention_bilstm_glove_2layer.pth"

TOKENIZER_PATH = r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\Tonedownlemma\GloVe_stack_BiLSTM_Attention\tokenizer_glove.pkl"

CONFIG_PATH = r"E:\Customer_Feedback_Sentiment_Analyzer\Saved\Tonedownlemma\GloVe_stack_BiLSTM_Attention\config_glove.pkl"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(CONFIG_PATH, "rb") as f:
    config = pickle.load(f)

vocab_size = config["vocab_size"]
embedding_dim = config["embedding_dim"]
hidden_dim = config["hidden_dim"]
max_len = config["max_len"]

# LOAD MODEL

model = AttentionBiLSTM(
    vocab_size,
    embedding_dim,
    hidden_dim
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

# PREDICTION

def predict_sentiment(text):

    text = preprocess_lstm(text)

    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    tensor = torch.tensor(
        padded,
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(
            output
        ).item()

    sentiment = (
        "Positive"
        if probability >= 0.5
        else "Negative"
    )

    confidence = (
        probability * 100
        if probability >= 0.5
        else (1 - probability) * 100
    )

    return sentiment, confidence, probability

# UI

review = st.text_area(
    "Enter Review",
    height=150
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        sentiment, confidence, probability = predict_sentiment(review)

        if sentiment == "Positive":
            st.success(
                f"😊 Positive Review\n\nConfidence: {confidence:.2f}%"
            )
        else:
            st.error(
                f"😞 Negative Review\n\nConfidence: {confidence:.2f}%"
            )

        st.write(
            f"Raw Probability: {probability:.4f}"
        )