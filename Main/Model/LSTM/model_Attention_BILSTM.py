#IMPORTS
import pandas as pd
import numpy as np

import os
import pickle

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# LOAD DATA
from google.colab import drive
drive.mount('/content/drive')

file_path = "/content/drive/MyDrive/sentiment_analyzer/cleaned_imdb_dataset_lemma.csv"
df = pd.read_csv(file_path)

X = df["review"]
y = df["sentiment"]

print("Data Loaded")


#TOKENIZATION
tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)

max_len = 200
X_pad = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

X_tensor = torch.tensor(X_pad, dtype=torch.long)

y = y.map({"positive": 1, "negative": 0})
y_tensor = torch.tensor(y.values, dtype=torch.float32)

print("Data prepared")


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor,
    test_size=0.2,
    random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)



# ATTENTION LSTM MODEL

class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
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

        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):

        embedded = self.embedding(x)

        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch, seq_len, hidden*2)

        # ATTENTION MECHANISM
        attn_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),    #The magic is where it is applied and what happens afterward.
            #Instead of throwing away 199 states: Attention looks at ALL hidden states.
            dim=1
        )  # (batch, seq_len)
        #softmax:
        """
        scores: [0.2, 0.1, 0.3, 3.5, 4.2]
        Softmax converts them into: [0.01, 0.01, 0.01, 0.31, 0.66]
        Now:31% importance,66% importance
        
        Review
        ↓
        BiLSTM
        ↓
        Score every hidden state
        ↓
        Compute importance weights
        ↓
        Weighted combination
        ↓
        Linear
        
        """

        context_vector = torch.sum(
            lstm_out * attn_weights.unsqueeze(-1),
            dim=1
        )  # (batch, hidden*2)
        #context_vector =Σ(weight × hidden_state) -> summation 

        out = self.dropout(context_vector)
        out = self.fc(out)

        return out


#MODEL INIT
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
hidden_dim = 256

model = AttentionLSTM(vocab_size, embedding_dim, hidden_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)


# TRAINING LOOP
epochs = 10

model.train()

for epoch in range(epochs):
    total_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# EVALUATION
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs).squeeze()
        preds = torch.sigmoid(outputs)

        predicted = (preds >= 0.5).float()

        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())

y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_labels).numpy()

acc = accuracy_score(y_true, y_pred)

print("\nFINAL ACCURACY:", acc)


# SAVE MODEL

SAVE_DIR = "/content/drive/MyDrive/sentiment_analyzer/saved_attention_bilstm"

os.makedirs(SAVE_DIR, exist_ok=True)

# Save model weights
torch.save(
    model.state_dict(),
    os.path.join(
        SAVE_DIR,
        "attention_bilstm.pth"
    )
)

# Save tokenizer
with open(
    os.path.join(
        SAVE_DIR,
        "tokenizer.pkl"
    ),
    "wb"
) as f:
    pickle.dump(tokenizer, f)

# Save config
config = {
    "vocab_size": vocab_size,
    "embedding_dim": embedding_dim,
    "hidden_dim": hidden_dim,
    "max_len": max_len
}

with open(
    os.path.join(
        SAVE_DIR,
        "config.pkl"
    ),
    "wb"
) as f:
    pickle.dump(config, f)

print("\nModel saved successfully!")
print("Location:", SAVE_DIR)


#  CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.show()
"""
Old Model(model_lstm_bi,BiLSTM):
Use only final hidden state
      ↓
Prediction

New Model:
Look at ALL hidden states
Decide which words are important
      ↓
Prediction

Review
 ↓
Tokenizer
 ↓
Integer Sequence
 ↓
Padding
 ↓
Embedding
 ↓
BiLSTM
 ↓
Attention
 ↓
Context Vector
 ↓
Dropout
 ↓
Linear Layer
 ↓
Sentiment
"""