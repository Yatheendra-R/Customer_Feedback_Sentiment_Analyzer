# GloVe 300d + 2-Layer BiLSTM + Masked Attention
# Sentiment Analysis 

import os
import pickle
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# MANUAL TEST CASES

def predict_text(text):

    model.eval()

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

    print("\n--------------------------------")
    print("Review:")
    print(text)

    print("Predicted Sentiment:", sentiment)

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        f"Raw Probability: {probability:.4f}"
    )



# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# GOOGLE DRIVE
from google.colab import drive
drive.mount('/content/drive')

# DATASET PATH
file_path = "/content/drive/MyDrive/sentiment_analyzer/cleaned_imdb_lstm.csv"

df = pd.read_csv(file_path)

X = df["review"]
y = df["sentiment"]

print("Dataset Size:", len(df))


# TOKENIZATION
tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)

max_len = 400

X_pad = pad_sequences(
    sequences,
    maxlen=max_len,
    padding="post",
    truncating="post"
)

X_tensor = torch.tensor(X_pad, dtype=torch.long)

# LABELS
y = y.map({
    "positive": 1,
    "negative": 0
})

y_tensor = torch.tensor(
    y.values,
    dtype=torch.float32
)
"""
# DOWNLOAD GLOVE
if not os.path.exists("glove.6B.300d.txt"):
    !wget http://nlp.stanford.edu/data/glove.6B.zip
    !unzip -o glove.6B.zip
"""
embedding_dim = 300

# LOAD GLOVE
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
            dtype="float32"
        )

        embeddings_index[word] = vector

print("Loaded GloVe:", len(embeddings_index))

# EMBEDDING MATRIX
vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros(
    (vocab_size, embedding_dim)
)

for word, idx in tokenizer.word_index.items():

    # Handle negation tokens
    if word.startswith("NOT_"):

        base_word = word[4:]      # NOT_good -> good

        vector = embeddings_index.get(base_word)

        if vector is not None:

            # opposite direction embedding
            embedding_matrix[idx] = -vector

    else:

        vector = embeddings_index.get(word)

        if vector is not None:

            embedding_matrix[idx] = vector

print("Embedding Matrix Ready")


# TRAIN TEST SPLIT
"""X_train, X_test, y_train, y_test = train_test_split(
    X_tensor,
    y_tensor,
    test_size=0.2,
    random_state=42,
    stratify=y_tensor

)"""

X_train, X_temp, y_train, y_temp = train_test_split(
    X_tensor,
    y_tensor,
    test_size=0.2,
    random_state=42,
    stratify=y_tensor
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
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
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)

# MODEL
class AttentionBiLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        embedding_matrix
    ):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            ),
            freeze=False,
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
        #hidden shape=>(4, 32, 256)  2 layers * 2 direction =4
        #Hidden state stores the final short-term state.

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

# MODEL INIT
hidden_dim = 256

model = AttentionBiLSTM(
    vocab_size,
    embedding_dim,
    hidden_dim,
    embedding_matrix
).to(device)

criterion = nn.BCEWithLogitsLoss() #BCELoss(sigmoid(output), target)
"""
Model Output (Logits)
      ↓
Sigmoid
      ↓
Probability (0 to 1)
      ↓
Binary Cross Entropy
      ↓
Loss Value
"""

optimizer = torch.optim.AdamW(
    model.parameters(), #contains all trainable weights:Embedding weights LSTM weights ,Attention weights ,FC weights
    lr=0.0005,
    weight_decay=1e-5 #This is a regularization term.
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  #This changes the learning rate automatically.
    optimizer,
    mode='max',
    factor=0.5,  #New LR=Old LR × 0.5
    patience=1  #Wait 1 epoch without improvement
)

# TRAINING
epochs = 8

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device) #(32,400)
        labels = labels.to(device)  #(32,)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(   #Gradient clipping.
            model.parameters(),
            5
        )

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss={total_loss:.4f}"
    )

# EVALUATION
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

y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_labels).numpy()

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\\nFINAL ACCURACY:", accuracy)

# CONFUSION MATRIX
cm = confusion_matrix(
    y_true,
    y_pred
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative","Positive"],
    yticklabels=["Negative","Positive"]
)

plt.title(
    "GloVe + Stacked BiLSTM + Masked Attention"
)

plt.show()

# SAVE MODEL
SAVE_PATH = "/content/drive/MyDrive/sentiment_analyzer/"

torch.save(
    model.state_dict(),
    SAVE_PATH + "attention_bilstm_glove_2layer.pth"
)

with open(
    SAVE_PATH + "tokenizer_glove.pkl",
    "wb"
) as f:
    pickle.dump(tokenizer, f)

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
    pickle.dump(config, f)

print("Saved successfully.")


# TEST REVIEWS
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

print("\n========================")
print("MANUAL TEST CASES")
print("========================")

for review in test_reviews:

    predict_text(review)
print(tokenizer.word_index.get("not"))

count = df["review"].str.contains(r"\bnot\b", case=False, na=False).sum()

print("Reviews containing NOT:", count)
print("Percentage:", count / len(df) * 100)