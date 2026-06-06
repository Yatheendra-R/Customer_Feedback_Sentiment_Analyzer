"""
By default now your TF-IDF probably uses: unigrams

Meaning: single words only

Example:

Sentence: "This movie is not good"

Unigrams become: ["movie", "not", "good"]

The Problem

TF-IDF sees:

"not"
"good"

as separate features.

But sentiment actually comes from: "not good"
together.



n-grams Do

Bigram (2-word combinations)

Now TF-IDF can also learn: ["not good"] as ONE feature.

That is extremely powerful for sentiment.

Example

Sentence: "The acting was very good"

Unigrams: ["acting", "very", "good"]

Bigrams: ["very good"]

Now model learns: "very good" is strongly positive.



"""

"""
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000,
    min_df=2,
    max_df=0.8,
    sublinear_tf=True
)

1. ngram_range (you already learned)
(1,2) or (1,3)
2. max_features (VERY important)

Limits vocabulary size:

max_features=5000

Why?

removes rare noisy words
speeds up training
improves generalization
3. min_df (removes rare words)
min_df=2

Meaning:

ignore words that appear in less than 2 documents

Removes:

typos
noise words
one-time tokens
4. max_df (removes too common words)
max_df=0.8

Meaning:

ignore words appearing in >80% documents

Removes:

useless common words like "movie", "film"
5. sublinear_tf (VERY powerful)
sublinear_tf=True

Instead of raw frequency:

tf = count

It uses:

tf = 1 + log(count)

Helps reduce bias toward long reviews.


| Change       | Benefit                      |
| ------------ | ---------------------------- |
| max_features | reduces noise                |
| min_df       | removes rare junk            |
| max_df       | removes common useless words |
| sublinear_tf | stabilizes long reviews      |
| ngram_range  | adds context                 |

"""

"""
What does smooth_idf=True mean?

First recall TF-IDF:

IDF = log(N / df)

Where:

N = total documents
df = number of documents containing the word
 Problem without smoothing

If a word appears in almost all documents, then:

df ≈ N

So:

log(N/N) = log(1) = 0

This can make some calculations unstable or too harsh.
What smoothing does

With:

smooth_idf=True

TF-IDF becomes:  IDF = log((1 + N) / (1 + df)) + 1

Why this helps

It prevents:
    division by zero issues
    extreme IDF values
    overly aggressive down-weighting

| Case        | Without smoothing   | With smoothing         |
| ----------- | ------------------- | ---------------------- |
| common word | may become 0 weight | small but non-zero     |
| rare word   | high weight         | controlled high weight |
| stability   | less stable         | more stable            |

"""