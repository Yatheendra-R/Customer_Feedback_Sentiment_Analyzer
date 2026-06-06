"""
Text
 ↓
Tokenization
 ↓
Word IDs
 ↓
Padding
 ↓
Embedding Layer
 ↓
LSTM
 ↓
Linear Layer
 ↓
Prediction
"""
"""
| Token   | Purpose       |
| ------- | ------------- |
| `<PAD>` | padding       |
| `<UNK>` | unknown words |
<PAD> is:artificial empty space ,not an actual word with meaning.
If the model learns embeddings for PAD:
    it may treat padding like real language
    introduce noise
    confuse sequence learning
<UNK> helps the model handle words that were never seen during training.
Without it: new/unseen words could break the encoding pipeline

max sequence length

Example:

MAX_LEN = 100
Meaning:

shorter sequences → padded
longer sequences → truncated 

Important Tradeoff

Small max length:
 faster
 may lose information

Large max length:
 preserves context
 slower training

Long sequences increase:

GPU/CPU memory consumption

which can:

slow training heavily
cause out-of-memory errors
crash notebooks/programs sometimes

Dataset & DataLoader

Why needed?

Because deep learning training needs:

batching
shuffling
efficient loading

instead of manually handling:

one sentence at a time

Dataset:

stores all samples

DataLoader:

gives batches during training

Example:

Batch 1 → 32 reviews
Batch 2 → 32 reviews
...

This makes training efficient.

Training with batches is better mainly because:

modern hardware (GPU/CPU) processes many samples in parallel
So batches make training:

much faster
more memory efficient
computationally optimized

Instead of:

1 review → update
1 review → update
1 review → update

we do:

32 reviews together → one update

This is more stable and efficient.
Another Important Advantage

Batches produce:

more stable gradient updates

because learning is based on:

multiple examples together

instead of one noisy example.
| Type          | Update Timing       |
| ------------- | ------------------- |
| Stochastic GD | after every sample  |
| Batch GD      | after full dataset  |
| Mini-batch GD | after small batches |


Problem Without Dataset/DataLoader

Suppose you manually do:

take sentence
convert to tensor
send to model
repeat thousands of times

Very inefficient and messy.

Dataset

Dataset stores:

(input, label) pairs

Example:

Encoded Review	Label
[1,2,3,4]	1
[5,6,7,0]	0

Think of Dataset as:

organized storage
DataLoader

DataLoader automatically:

creates batches
shuffles data
feeds batches during training

Example:

Batch 1:
32 reviews

Batch 2:
32 reviews

Why Shuffling Matters

Suppose dataset order is:

positive
positive
positive
negative
negative
negative

Without shuffling:

model may learn biased patterns
training becomes unstable

So DataLoader often uses:

shuffle=True

Partly related, but the more direct reason is:

Shuffling prevents the model from learning biased patterns from the data order.

Example:

positive
positive
positive
negative
negative
negative

If batches come in this order:

model temporarily sees only positives
then only negatives

Training becomes unstable and biased.

After Shuffling
positive
negative
positive
negative
...

Now batches become more representative.

This helps:
 smoother learning
 better gradient updates
 better generalization

And indirectly:

can help reduce overfitting/bias

So your answer was directionally connected.


The final hidden state acts like:

compressed understanding of the whole sentence

So instead of classifying every word individually, we classify:

the sentence representation

That representation is then sent to:

Linear Layer → sentiment prediction
Example Intuition

Sentence:

"I thought the movie would be amazing but it was not good"

As LSTM reads words sequentially:

memory updates continuously
context accumulates

Final hidden state tries to capture:

overall sentiment/context
Important Distinction

LSTM internally produces:

hidden state at every word

Example:

Word	Hidden State
I	h₁
thought	h₂
movie	h₃
...	...
good	hₙ

For sentiment classification we usually use:

hₙ (final hidden state)
Why Not Use All Outputs?

Possible, but:

more computation
more complexity
not necessary for beginner sentiment classification

Using:

final hidden state

is the standard/simple approach.

Simple Intuition
last hidden state ≈ summary of the sequence

That summary goes into:

Linear Layer

for classification.

Important NLP Tensor Shape

Batch example:

(batch_size, sequence_length)

Example:

(32, 100)

Meaning:

32 reviews
each padded to length 100

Important Tensor Shape

This becomes very important in PyTorch.

LSTM input shape usually:

(batch_size, sequence_length, embedding_dim)

Example:

(32, 100, 128)

Meaning:

32 sentences
each sentence length = 100
each word embedding size = 128

What Does 128 Mean?

Each word is represented using:

128 learned numeric features

Example:

love → [0.12, -0.55, 0.91, ..., 128 values]

These features are NOT manually defined.

The model automatically learns patterns like:

positivity
negativity
similarity
grammatical behavior
semantic relationships

during training.

Important Intuition

Embedding dimension is like:

how much space the model has to store word meaning

nn.Embedding(vocab_size, embedding_dim)

Example:

nn.Embedding(10000, 100)

Meaning:

vocabulary size = 10,000
each word learns 100 features



Embedding layer:

understands individual words

LSTM hidden state:

understands sequence/context


Embedding Dimension

Represents:

word meaning representation size

Example:

128 features per word
Hidden Size

Represents:

LSTM memory capacity

Example:

how much contextual information the LSTM can store

LSTM Hidden State

Main job:

learn sequence/context understanding

Example:

"not good" → negative

because order/context matters.


Epochs

Very important in training.

Suppose:

dataset = 10,000 reviews

One:

epoch

means:

model sees entire training dataset once

| Term      | Meaning             |
| --------- | ------------------- |
| Epoch     | full dataset pass   |
| Batch     | small chunk of data |
| Iteration | one batch update    |

But Too Many Epochs Cause
Overfitting

Model starts:

memorizing training data

instead of generalizing.

Too many epochs can cause:

model memorization instead of generalization

Meaning:

training accuracy becomes very high
but performance on new reviews worsens

solution of overfitting
Common Solutions
 Fewer Epochs

Simplest solution.

 Dropout

Randomly disables some neurons during training.

Helps prevent:

over-dependence on specific patterns
 More Data

Usually best solution.

 Regularization

Advanced topic for later.

split data into:
| Split      | Purpose                |
| ---------- | ---------------------- |
| Training   | learn patterns         |
| Validation | tune/check performance |
| Test       | final evaluation       |

Why Validation Matters

Without validation:

you cannot detect overfitting properly

because training accuracy alone can be misleading.

Example
Training Accuracy = 99%
Validation Accuracy = 78%

Danger sign:

overfitting


Loss Function

Very important.

The model predicts:

0.91
0.12
0.67

But:

how does model know if prediction is bad?

Answer:

Loss Function
In Binary Sentiment Classification

Most common:

Binary Cross Entropy Loss (BCE Loss)

It measures:

how wrong predictions are

Goal of Training

Training tries to:

minimize loss

meaning:

make predictions closer to correct labels
PyTorch Version

Usually:

nn.BCEWithLogitsLoss()

Very common for:

binary classification
Important Clarification

Loss:

≠ accuracy

Accuracy:

how many predictions correct

Loss:

how wrong/confident predictions are

Loss measures how wrong the predictions are (including confidence), while accuracy measures how many predictions are correct.

Accuracy → count of correct predictions
Loss → quality/confidence of predictions


Optimizer

Question:

How does model reduce loss?

Answer:

optimizer
Most Common Optimizer

For your project:

Adam

PyTorch:

torch.optim.Adam()
Optimizer Intuition

Optimizer:

updates model weights to reduce loss

Prediction
   ↓
Calculate Loss
   ↓
Compute Gradients
   ↓
Optimizer Updates Weights
   ↓
Better Prediction

for many batches and epochs

loss = mistakes
optimizer = correction mechanism

Adam

Why popular?

fast
stable
works well in practice
beginner-friendly


Gradient

A gradient basically tells:which direction should weights move to reduce loss
Goal of training: reach lowest loss possible
Gradient tells:
    which direction is downhill
    how steep the slope is


Example Intuition

Suppose model weight:

w = 0.5

Current loss:

10

Gradient may indicate:

increase w

or

decrease w

to reduce loss.

Optimizer uses this information.


Compute Gradients

During: backpropagation

PyTorch calculates: how much each weight contributed to the loss

These values are: gradients
Forward Pass
(model prediction)
        ↓
Calculate Loss
        ↓
Backward Pass
(compute gradients)
        ↓
Optimizer updates weights

Formula Intuition

Gradient is mathematically: ∂Loss/∂Weight

Meaning: how much loss changes when weight changes

| Gradient Value | Meaning                      |
| -------------- | ---------------------------- |
| large          | weight strongly affects loss |
| small          | weight has little effect     |
| positive       | move weight downward         |
| negative       | move weight upward           |

The gradient tells the optimizer which direction the weights should move to reduce loss.
gradient = direction of steepest increase
So the optimizer moves: opposite to the gradient
to go downhill toward: lower loss
"""
"""
Learning Rate

How big should weight updates be? -> learning rate

Suppose optimizer wants to update weights.

Very Small Learning Rate->tiny steps downhill

    stable
    very slow training

Very Large Learning Rate -> huge jumps downhill

    may overshoot
    unstable training
    loss may explode

Mountain Analogy Again

Learning rate controls: step size while going downhill

Typical Values Common:
   0.1
   0.01
   0.001

For Adam optimizer: 0.001 is a very common good starting point.

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
"""
