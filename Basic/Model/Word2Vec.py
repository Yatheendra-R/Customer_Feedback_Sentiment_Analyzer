"""
TF-IDF

Represents: importance of words in a document

Characteristics:

    sparse vectors
    mostly counting/statistics
    no semantic understanding

good and excellent → completely different
"""

"""
Word Embeddings

Represent: meaning/semantic relationships of words

Characteristics:

    dense vectors
    learned from context
    semantically similar words become close

Example:

good ≈ excellent
king ≈ queen

So the big shift is:

    TF-IDF → “How important is this word?”
    Embeddings → “What does this word mean?”
"""

"""
Word2Vec

This is the first major embedding model you should learn.

Core idea: words appearing in similar contexts have similar meanings

Example:

The cat drinks milk
The dog drinks milk

Word2Vec learns:  cat ≈ dog

because their contexts are similar.

What Actually Gets Learned? Initially vectors are random.

Example (random start):

Word	Vector
cat	   [0.2, -0.5, 0.1]
dog	   [0.8, 0.1, -0.2]

After training:

Word	Vector
cat	  [0.91, 0.22, 0.74]
dog	  [0.89, 0.25, 0.71]

Now:

cat ≈ dog
because contexts were similar.
"""

"""
Two Word2Vec approaches:

CBOW (Continuous Bag of Words)

Idea: Predict the missing word from surrounding words.

Example: "The ___ drinks milk"

Context:
    the
    drinks
    milk

Predict: cat

So:  context → target word

Skip-gram

Idea:  Use one word to predict surrounding words.

Example:

Input: cat

Predict:

    the
    drinks
    milk

So: Target word → context

CBOW:
    faster
    works well for common words

Skip-gram:
    better for rare words
    usually better embeddings
"""

"""
An important limitation of Word2Vec:

Problem Each word gets only ONE vector.

So:  bank → same embedding always

But:

    bank (money)
    bank (river)

have different meanings.

Word2Vec cannot fully understand this contextual meaning.

This limitation eventually leads toward:

    RNN/LSTM
    Attention
    Transformers
    BERT

"""
"""
Example:

Context Window
Suppose window size = 1

Sentence: cat drinks milk

For the word: drinks

context words are:
    cat
    milk

So the model learns:

    drinks ↔ cat
    drinks ↔ milk

After seeing millions of examples, embeddings start forming semantic meaning.

CBOW Example

Sentence: cat drinks milk

Input:
cat
milk

Predict: drinks

So: (context words) → target word

Skip-gram Example

Sentence: cat drinks milk

Input: drinks

Predict:
    cat
    milk

So: (target word) → surrounding words

"""
"""
Famous Word2Vec Property

The model learns relationships:

king - man + woman ≈ queen

Meaning:
    “king” and “queen” are related similarly
    gender relationship gets encoded mathematically

This shocked researchers when Word2Vec came out.
"""
"""
Word2Vec understands:
 semantic similarity

But still struggles with:
 sentence flow
 long context
 dynamic meaning based on sentence

One Very Important Practical Point

Word2Vec itself is NOT usually the final classifier.

It is:

a feature representation method

Then embeddings are fed into:

RNN
LSTM
GRU
CNN
small neural networks

Example:

Word2Vec → LSTM → sentiment prediction
"""
"""

TF-IDF becomes Sparse

Let’s take a tiny vocabulary: ["good", "bad", "movie", "food", "excellent"]

Now sentence: "good movie"

TF-IDF vector becomes something like:

[0.8, 0, 0.6, 0, 0]

Why many zeros?
    Because:
        the sentence only contains a few words
        most vocabulary words are absent

Real NLP vocabularies can have: 10,000 → 100,000+ words

But one sentence may contain only: 10–20 words

So most positions become zero.

That naturally creates: sparse vectors
 
Why Embeddings Become Dense

Word2Vec does NOT allocate:  one dimension per vocabulary word

Instead it learns: small compressed feature representations

Example:
Instead of:

10,000 dimensions

it may learn:

100 or 300 dimensions

Each dimension captures hidden semantic patterns like:

positivity
gender relation
animal-like behavior
royalty
tense
etc.

Since every dimension stores learned information:

most values become useful numbers

So embeddings are dense.

Important Analogy

TF-IDF:

Huge checklist of words

Embeddings:

Compressed meaning representation

That’s the real difference.
"""