# IMPORTS
import pandas as pd
import numpy as np
import pickle
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Imported Successfully")


#DEVICE SETUP
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

"""
#DOWNLOAD GLOVE
if not os.path.exists("glove.6B.300d.txt"):

    !wget http://nlp.stanford.edu/data/glove.6B.zip

    !unzip glove.6B.zip

print("GloVe Ready!")
"""

#  LOAD DATA
from google.colab import drive

drive.mount('/content/drive')

file_path = "/content/drive/MyDrive/sentiment_analyzer/cleaned_imdb_lstm.csv"

df = pd.read_csv(file_path)

X = df["review"]
y = df["sentiment"]

print("Dataset Loaded!")


#TOKENIZATION
tokenizer = Tokenizer(
    oov_token="<UNK>"
)

tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)

max_len = 200

X_pad = pad_sequences(
    sequences,
    maxlen=max_len,
    padding="post",
    truncating="post"
)

X_tensor = torch.tensor(
    X_pad,
    dtype=torch.long
)

print("Tokenization Completed!")


#LABEL ENCODING
y = y.map({
    "positive": 1,
    "negative": 0
})

y_tensor = torch.tensor(
    y.values,
    dtype=torch.float32
)

print("Labels Encoded!")


#LOAD GLOVE EMBEDDINGS
embedding_dim = 300

embeddings_index = {}

with open(
    "glove.6B.300d.txt",
    encoding="utf8"
) as f:

    for line in f:

        values = line.split()

        word = values[0]

        vector = np.asarray(
            values[1:],
            dtype='float32'
        )

        embeddings_index[word] = vector

print(
    "Loaded GloVe vectors:",
    len(embeddings_index)
)


# CREATE EMBEDDING MATRIX
vocab_size = len(
    tokenizer.word_index
) + 1

embedding_matrix = np.zeros(
    (vocab_size, embedding_dim)
)

for word, index in tokenizer.word_index.items():

    if index >= vocab_size:
        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector

print("Embedding Matrix Created!")


#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor,
    y_tensor,
    test_size=0.2,
    random_state=42
)

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64
)

print("Train/Test Split Done!")


#ATTENTION MODEL
class AttentionBiLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        embedding_matrix
    ):
        super().__init__()

        # PRETRAINED GLOVE
        self.embedding = nn.Embedding.from_pretrained(

            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            ),

            freeze=True,
            padding_idx=0
        )

        self.lstm = nn.LSTM(

            embedding_dim,

            hidden_dim,

            batch_first=True,

            bidirectional=True
        )

        # ATTENTION
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

        embedded = self.embedding(x)

        lstm_out, (hidden, cell) = self.lstm(
            embedded
        )

        # ATTENTION WEIGHTS
        attn_weights = torch.softmax(

            self.attention(lstm_out).squeeze(-1),

            dim=1
        )

        # CONTEXT VECTOR
        context_vector = torch.sum(

            lstm_out * attn_weights.unsqueeze(-1),

            dim=1
        )

        out = self.dropout(
            context_vector
        )

        out = self.fc(out)

        return out


# MODEL INIT
hidden_dim = 256

model = AttentionBiLSTM(

    vocab_size,

    embedding_dim,

    hidden_dim,

    embedding_matrix

).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.0007
)

print("Model Initialized!")


#TRAINING
epochs = 10

model.train()

for epoch in range(epochs):

    total_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5
        )

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}, "
        f"Loss: {total_loss:.4f}"
    )


#EVALUATION
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs).squeeze()

        preds = torch.sigmoid(outputs)

        predicted = (
            preds >= 0.5
        ).float()

        all_preds.append(
            predicted.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

y_pred = torch.cat(
    all_preds
).numpy()

y_true = torch.cat(
    all_labels
).numpy()

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\nFINAL ACCURACY:", accuracy)


# CONFUSION MATRIX
cm = confusion_matrix(
    y_true,
    y_pred
)

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=[
        "Negative",
        "Positive"
    ],

    yticklabels=[
        "Negative",
        "Positive"
    ]
)

plt.title(
    "Attention BiLSTM + GloVe"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


#SAVE EVERYTHING
SAVE_PATH = "/content/drive/MyDrive/sentiment_analyzer/"

# SAVE MODEL
torch.save(

    model.state_dict(),

    SAVE_PATH + "attention_bilstm_glove.pth"
)

print("Model Saved!")

# SAVE TOKENIZER
with open(
    SAVE_PATH + "tokenizer_glove.pkl",
    "wb"
) as f:

    pickle.dump(
        tokenizer,
        f
    )

print("Tokenizer Saved!")

# SAVE CONFIG
config = {

    "vocab_size": vocab_size,

    "embedding_dim": embedding_dim,

    "hidden_dim": hidden_dim,

    "max_len": max_len
}

with open(
    SAVE_PATH + "config_glove.pkl",
    "wb"
) as f:

    pickle.dump(
        config,
        f
    )

print("Config Saved!")

print("\nALL FILES SAVED TO GOOGLE DRIVE!")