#IMPORTS
import pandas as pd
import numpy as np

import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("Imported Successfully")

# DEVICE SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# MODEL (IMPROVED LSTM)
class SentimentLSTM(nn.Module):
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

        self.dropout = nn.Dropout(0.5)  

        self.fc = nn.Linear(hidden_dim * 2, 1)  
    
        #Forward Features (256) +Backward Features (256)
        """
        (num_layers * directions,batch_size,hidden_dim)
        1 layer
        2 directions
        (2,64,256)<=hidden.shape
        hidden[0]=Forward final hidden state hidden[0].shape=>(64,256)
        hidden[1]=Backward final hidden state hidden[1].shape=>(64,256)
        """
        

            
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)

        # concatenate forward + backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  
        #(64,512)
        #Forward Summary: [256 features] + Backward Summary: [256 features] => combine: [512 features]
        
        out = self.dropout(hidden)
        out = self.fc(out)

        return out


# LOAD DATA
from google.colab import drive
drive.mount('/content/drive')

file_path = "/content/drive/MyDrive/sentiment_analyzer/cleaned_imdb_dataset_lemma.csv"
df = pd.read_csv(file_path)

print(df.head())

review_col = "review"
sentiment_col = "sentiment"

X = df[review_col]
y = df[sentiment_col]


#TOKENIZATION

tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)

max_len = 200

padded_sequences = pad_sequences(
    sequences,
    maxlen=max_len,
    padding="post",
    truncating="post"
)

X = torch.tensor(padded_sequences, dtype=torch.long)

# LABEL ENCODING
y = y.map({"positive": 1, "negative": 0})
y = torch.tensor(y.values, dtype=torch.float32)

print("Data prepared")


#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


#  MODEL INIT
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
hidden_dim = 256

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)  # 🔥 slightly lower LR


# TRAINING LOOP

num_epochs = 10  # 10–12 is optimal for IMDB

model.train()

for epoch in range(num_epochs):
    epoch_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()

        loss = criterion(outputs, labels)

        loss.backward()

        # gradient stability fix
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        """
        LSTMs sometimes suffer: Gradient Explosion
        Example: Gradient = 5000
        Huge update. Training becomes unstable.
        """
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")


#  EVALUATION (BATCH SAFE)
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

accuracy = accuracy_score(y_true, y_pred)

print("\nFINAL ACCURACY:", accuracy)

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


# CONFUSION MATRIX
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
Let's use a very small example.

Review:

a b c d

Assume:

hidden_dim = 3

(I'm using 3 instead of 256 so we can visualize it.)

Step 1: Process word "a"

LSTM reads:

a

and creates:

h1 = [0.2, 0.7, -0.1]

This vector represents everything the LSTM knows after reading:

a
Step 2: Process word "b"

Now LSTM has seen:

a b

It updates its memory and produces:

h2 = [0.5, 0.3, 0.8]

Notice:

h2 ≠ representation of only b

Instead:

h2 = representation of "a b"
Step 3: Process word "c"

Now LSTM has seen:

a b c

and produces:

h3 = [0.9, -0.2, 0.4]

This represents:

"a b c"
Step 4: Process word "d"

Now LSTM has seen:

a b c d

and produces:

h4 = [0.1, 0.8, 0.6]

This represents:

"a b c d"

(the entire review)

What is output?

output stores ALL hidden states:

output =
[
 h1,
 h2,
 h3,
 h4
]

Numerically:

output =
[
 [0.2, 0.7, -0.1],   # after a
 [0.5, 0.3, 0.8],    # after a b
 [0.9,-0.2, 0.4],    # after a b c
 [0.1, 0.8, 0.6]     # after a b c d
]

Shape:

(seq_len, hidden_dim)

(4,3)

In your model:

(200,256)

for one review.

What is hidden?

hidden contains ONLY the last hidden state:

hidden = h4

which is:

hidden =
[0.1, 0.8, 0.6]

This is the LSTM's final summary after reading:

a b c d
Very Important

Many beginners think:

h1 = meaning of a
h2 = meaning of b
h3 = meaning of c
h4 = meaning of d

 Wrong.

Actually:

h1 = meaning of "a"

h2 = meaning of "a b"

h3 = meaning of "a b c"

h4 = meaning of "a b c d"

Each hidden state contains information about everything seen so far.
"""