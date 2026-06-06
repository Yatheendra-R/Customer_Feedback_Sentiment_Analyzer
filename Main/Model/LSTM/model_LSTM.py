import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn #nn neural network
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
print("Imported")
"""
CSV Dataset
 ↓
Tokenizer
 ↓
Word Index
 ↓
Integer Sequences
 ↓
Padding / Truncation
 ↓
PyTorch Tensors
 ↓
Train-Test Split
 ↓
Dataset / DataLoader
 ↓
Embedding + LSTM
 ↓
Prediction
"""

class SentimentLSTM(nn.Module): #nn.Module is the base class for neural networks
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super().__init__() #nn.Module constructor
        self.embedding = nn.Embedding(vocab_size, 
                                      embedding_dim,    
                                      padding_idx=0 #PAD token should not learn semantic meaning
                    )
        #Embedding output shape: (batch_size, seq_len, embedding_dim) 
        # Example: (32, 100, 128)

        #embedding vectors as input.So input size must equal: embedding_dim
        #embedding_dim = 128 Then each word vector has 128 features
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,batch_first=True)
        #Hidden Dimension hidden_dim  controls: LSTM memory/context capacity
        #Example: hidden_dim = 256
        #Then final hidden state shape becomes: (batch_size, 256)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)

        hidden = hidden.squeeze(0)

        out = self.fc(hidden)
        return out


df=pd.read_csv(r"E:\Internship for agilisium\sentiment_analyzer\Main\Data\cleaned_imdb_dataset_lemma.csv")
print(df.columns)
review=df.columns[0]
sentiment=df.columns[1]
print(review)
print(sentiment)
X=df[review]
y=df[sentiment]
tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(X)

sequences =tokenizer.texts_to_sequences(X)
#print(sequences)
padded_sequences=pad_sequences(sequences, maxlen=200,
                        padding ="post",truncating="post")
#print("After padding:\n",padded_sequences)
X = torch.tensor(padded_sequences,dtype=torch.long)
print(X.shape)
print(y.head())
print(y.dtype)
y = y.map({
    "positive": 1,
    "negative": 0
})
print("converted")
y = torch.tensor(y.values,dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
vocab_size = len(tokenizer.word_index) + 1
#vocab_size=10000 #10,000 words
embedding_dim = 128 #each word learns 128 features
hidden_dim = 256
model = SentimentLSTM(
    vocab_size,
    embedding_dim,
    hidden_dim
)
criterion = nn.BCEWithLogitsLoss()  #BCEWithLogitsLoss() internally handles: sigmoid activation
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
#model.parameters() Tells optimizer: which weights should be updated

model.train()
"""
Some layers behave differently during:
    training
    evaluation

Examples:
    Dropout
    BatchNorm
"""
num_epochs=3
for epoch in range(num_epochs): #one epoch = full dataset pass
    epoch_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad() #accumulates gradients by default , to clear old gradients before new batch computation
        outputs = model(inputs) #generate predictions using forward 
        #Because model output shape may be: (32, 1) 
        #but labels shape is: (32,) So: squeeze() removes extra dimension.
        loss = criterion(outputs.squeeze(), labels)
        epoch_loss += loss.item()
        loss.backward()
        #This computes: gradients for all trainable weights.
        optimizer.step()
        #Optimizer uses gradients to: adjust weights toward lower loss
        """
        Zero Gradients
        ↓
        Forward Pass
        ↓
        Loss Calculation
        ↓
        Backward Pass
        ↓
        Weight Update
        """
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
model.eval()
with torch.no_grad():   
    """
    we do NOT need gradients
    because: weights are not updating

    This:
    saves memory
    speeds evaluation
    avoids unnecessary computation
    """
    outputs = model(X_test)
    predictions = torch.sigmoid(outputs)
    predicted_classes = (predictions >= 0.5).float()
    accuracy = (
        predicted_classes.squeeze() == y_test
    ).float().mean()
    print("Accuracy:", accuracy.item())

y_true = y_test.cpu().numpy()
y_pred = predicted_classes.squeeze().cpu().numpy()

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
Raw Reviews
    ↓
Tokenizer
    ↓
Integer Encoding
    ↓
Padding
    ↓
Train/Test Split
    ↓
DataLoader
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Linear Layer
    ↓
Prediction
    ↓
Accuracy + Confusion Matrix
"""