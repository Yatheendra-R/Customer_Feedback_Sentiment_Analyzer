"""
Full Line
self.embedding = nn.Embedding(vocab_size, embedding_dim)

Let’s break it fully.

 self

self means:

current object of the class

Example:

model = SentimentLSTM()

Inside the class:

self refers to model
Why Needed?

Because every model object should store:

embedding layer
LSTM layer
linear layer

inside itself.

 self.embedding

This means:

create an attribute named embedding inside the model object

Example later:

model.embedding

accesses that layer.

Simple Analogy

Like:

self.name = "Yatheendra"

stores:

name attribute inside object

Similarly:

self.embedding

stores:

embedding layer inside model
 nn

From:

import torch.nn as nn

nn is just:

shortcut name (alias)

for:

torch.nn
 nn.Embedding

This means:

Embedding layer class provided by PyTorch

PyTorch already built it for us.

You are creating an object of that class.

Like Creating Any Object

Example:

car = Car()

Similarly:

embedding = nn.Embedding()
 Embedding

Embedding layer:

converts word IDs into dense vectors

Example:

15 → [0.21, -0.5, ...]

Internally:

it stores learnable embedding matrix
Internally It Looks Like

If:

vocab_size = 10000
embedding_dim = 128

Embedding matrix shape becomes:

(10000, 128)

Meaning:

10,000 rows
each row = one word vector
Final Meaning of Full Line
self.embedding = nn.Embedding(vocab_size, embedding_dim)

means:

Create an embedding layer and store it inside this model object.

VERY Important OOP + DL Connection

You are now seeing:

PyTorch = OOP + tensors + deep learning

This is why understanding classes helps a LOT in PyTorch.

self.embedding = nn.Embedding(vocab_size, embedding_dim)

Embedding is a class provided by torch.nn (aliased as nn).
We create an object(instance) of this class and store it as an attribute named embedding inside the model object using self.
"""
"""
torch

Main PyTorch library.

Contains:

tensors
GPU operations
autograd
optimizers
neural network modules
etc.
torch.nn

Submodule inside PyTorch for:

neural network components

Contains:

Embedding
LSTM
Linear
Conv2d
loss functions
etc.
"""
