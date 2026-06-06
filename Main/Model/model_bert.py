#INSTALL DEPENDENCIES
!pip install transformers datasets -q


#IMPORTS
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

import seaborn as sns
import matplotlib.pyplot as plt


#DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# LOAD DATA (GOOGLE DRIVE)
from google.colab import drive
drive.mount('/content/drive')

file_path = "/content/drive/MyDrive/sentiment_analyzer/cleaned_imdb_dataset_lemma.csv"
df = pd.read_csv(file_path)

df = df[['review', 'sentiment']]

# labels
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})


#TRAIN-TEST SPLIT
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'].values,
    df['label'].values,
    test_size=0.2,
    random_state=42
)



# TOKENIZER
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True,
    max_length=200
)

test_encodings = tokenizer(
    list(test_texts),
    truncation=True,
    padding=True,
    max_length=200
)


#DATASET CLASS
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)


#  MODEL
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
).to(device)


#  TRAINING ARGS
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",

    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    weight_decay=0.01,
    logging_steps=100,

    load_best_model_at_end=True,
    report_to="none"
)


#  TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)


#  TRAIN
trainer.train()


#  PREDICTIONS
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

accuracy = accuracy_score(test_labels, y_pred)
print("\nFINAL BERT ACCURACY:", accuracy)


# CONFUSION MATRIX
cm = confusion_matrix(test_labels, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.show()