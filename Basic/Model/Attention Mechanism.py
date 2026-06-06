"""
Classical NLP (TF-IDF Era)

You usually do heavy preprocessing:

  lowercase
  stopword removal
  stemming/lemmatization
  manual feature engineering

Because models are weak at understanding language structure.

Example:

  TF-IDF only sees tokens/statistics
  Deep Learning Era (LSTM/BERT)

Models learn patterns automatically.

So aggressive preprocessing can actually hurt.

Example: "I am not happy"

If you remove stopwords:  happy

Sentiment becomes wrong.

So with LSTM/BERT: removing “not” is dangerous.

Modern NLP Usually Keeps More Information

Instead of heavy cleaning, people often only:
  lowercase (sometimes)
  remove HTML
  remove weird symbols/noise
  tokenize text

But they KEEP:
  stopwords
  sentence structure
  grammar flow

because sequence models need context.

Very Important Difference
TF-IDF Models

Need:
  clean structured features
  LSTM/BERT

Need: natural language context

More traditional model → more preprocessing
More advanced neural model → less preprocessing
"""

"""
Attention Mechanism

The revolutionary idea: Instead of remembering everything, directly focus on important words.

Attention Mechanism

This changed NLP history.

Problem: Even LSTM must compress everything into memory.

Attention says:
    Why not directly look at important words when needed?

That idea eventually becomes: Transformer → BERT → GPT


LSTM Bottleneck

Even though LSTM is better than RNN: 
  long sequences still difficult
  memory compression still difficult

Especially:

  translation
  long documents
  question answering
"""

"""
Attention Idea (Revolutionary)

Instead of: remember everything

the model says:  when processing a word,directly look at important previous words

Example
  Sentence: "The movie was not good"

While processing: good

attention allows the model to directly focus on: not
instead of hoping memory preserved it.



Human Analogy

  When humans read: The movie was not good
  you naturally focus strongly on: not

Attention tries to mimic this behavior.

Why Attention Changed NLP

It solved many issues:
  long-range dependencies
  better context handling
  parallelization improvements
  stronger language understanding

This eventually led to:

Transformers From Attention → Transformer
Transformer -> This became the foundation of modern NLP.
"""
"""
Big Difference
RNN/LSTM

Process words: one-by-one sequentially

Problem:

  slower
  hard to parallelize
  long memory issues
  Transformer

Processes: all words simultaneously

using: Self-Attention

This was revolutionary.

Self-Attention Intuition

Sentence: "The movie was not good"

When understanding: good

the model can directly “look at”:

  not
  movie
  was
and assign importance scores.

Why Transformers Became Dominant

They:
  handle long context better
  train massively in parallel
  scale extremely well
  learn rich contextual meaning

That's why almost all modern NLP now uses transformers.

Very Important Concept

Unlike Word2Vec: bank → one fixed vector

Transformers create:

contextual embeddings

So:
  river bank
  money bank

get DIFFERENT meanings.

This is a massive leap in NLP capability.

Attention solved the problem of losing important information in long sequences by allowing
the model to directly focus on relevant words instead of relying only on one memory state.
"""

"""
Simplified Evolution Chain
TF-IDF
  ↓
Word2Vec
  ↓
RNN
  ↓
LSTM
  ↓
Attention
  ↓
Transformer
  ↓
BERT / GPT  


"""
"""
Embedding layer converts word IDs into dense semantic vectors that neural networks can learn from.

So What Does Embedding Layer Do?

It converts:

integer IDs → dense meaningful vectors

Example:

love → [0.91, 0.22, -0.5]
movie → [0.13, 0.77, 0.41]

These vectors are:

learned during training
semantically meaningful
"""