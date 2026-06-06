from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences   
import torch

sentences = [
    "i love this movie",
    "i hate this movie",
    "this move is not so great"
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
print(tokenizer.word_index)
"""
The IDs are usually assigned based on: word frequency
More frequent words often get: smaller IDs
"""

print(tokenizer.texts_to_sequences(["i love this movie"]))

"""
By default: padding happens at the beginning

called: pre-padding
People often prefer: padding='post'
pad_sequences(sequences, padding='post')
"""
sequences =tokenizer.texts_to_sequences(sentences)
print(sequences)
padded_sequences=pad_sequences(sequences, maxlen=5,
                        padding ="post",truncating="post")
print("After padding:\n",sentences)
X = torch.tensor(padded_sequences)
print(padded_sequences[2][2])
print(X.shape)
"""
truncating='post'

Removes words from end.

truncating='pre'

Removes words from beginning.
"""