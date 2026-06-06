#  IMPORTS
import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences

# PAGE CONFIG
st.set_page_config(
    page_title="Attention BiLSTM Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Attention BiLSTM Sentiment Analyzer")

st.markdown(
    """
This model uses:

 Stack BiLSTM  
 Attention Mechanism  
 Deep Learning Sentiment Analysis
 Glove 
no negation

"""
)

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#  MODEL CLASS
class AttentionLSTM(nn.Module):

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

        self.attention = nn.Linear(
            hidden_dim * 2,
            1
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, x):

        embedded = self.embedding(x)

        lstm_out, (hidden, cell) = self.lstm(
            embedded
        )

        # ATTENTION
        attn_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),
            dim=1
        )

        context_vector = torch.sum(
            lstm_out * attn_weights.unsqueeze(-1),
            dim=1
        )

        out = self.dropout(context_vector)

        out = self.fc(out)

        return out



#  PATHS

MODEL_PATH = r"E:\Internship for agilisium\Saved\Tonedownlemma\GloVe_stack_BiLSTM_Attention\attention_bilstm_glove_2layer.pth"

TOKENIZER_PATH = r"E:\Internship for agilisium\Saved\Tonedownlemma\GloVe_stack_BiLSTM_Attention\tokenizer_glove.pkl"

CONFIG_PATH = r"E:\Internship for agilisium\Saved\Tonedownlemma\GloVe_stack_BiLSTM_Attention\config_glove.pkl"


# LOAD CONFIG
with open(CONFIG_PATH, "rb") as f:

    config = pickle.load(f)

vocab_size = config["vocab_size"]

embedding_dim = config["embedding_dim"]

hidden_dim = config["hidden_dim"]

max_len = config["max_len"]


#LOAD TOKENIZER
with open(TOKENIZER_PATH, "rb") as f:

    tokenizer = pickle.load(f)


#LOAD MODEL
model = AttentionLSTM(
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


# PREPROCESSING
def preprocess(text):

    text = text.lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


#  PREDICTION
def predict_sentiment(text):

    cleaned_text = preprocess(text)

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    input_tensor = torch.tensor(
        padded,
        dtype=torch.long
    ).to(device)

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


#  USER INPUT
text = st.text_area(
    "Enter Review",
    height=180,
    placeholder="Type your review here..."
)


#  BUTTON
if st.button("Predict Sentiment"):

    if text.strip() == "":

        st.warning(
            "Please enter some text."
        )

    else:

        sentiment, confidence, prob, cleaned = predict_sentiment(text)

        # Processed text
        st.subheader("Processed Text")

        st.code(cleaned)

        # Prediction
        st.subheader("Prediction")

        if sentiment == "Positive":

            st.success(
                f"😊 Positive Sentiment"
            )

        else:

            st.error(
                f"😞 Negative Sentiment"
            )

        # Confidence
        st.subheader("Confidence")

        st.write(
            f"{confidence:.2f}%"
        )

        # Probability bar
        st.progress(float(confidence / 100))

        # Raw probability
        st.subheader("Raw Probability")

        st.write(
            f"{prob:.4f}"
        )


# FOOTER
st.markdown("---")

st.caption(
    "Built using PyTorch + Streamlit + Attention BiLSTM"
)