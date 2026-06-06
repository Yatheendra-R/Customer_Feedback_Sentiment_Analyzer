"""
Why RNN Was Needed

Word2Vec understands:
 word meaning

But it still struggles with:
 sentence flow
 long context
 dynamic meaning

Example:

"I am not happy"

The word:

not

changes the sentiment completely.

Classical models often struggle with this relationship.


RNN Core Idea

RNN reads words:

one by one in sequence

while carrying memory forward.

Simple visualization:

word1 → word2 → word3 → ...
        memory flows forward


Hidden State (Memory)

RNN keeps an internal memory called: hidden state

Very simplified idea:
ht =f(ht-1,xt)

Meaning: new memory = old memory + current word

RNN continuously updates memory while reading words.

Why RNN Is Better Than TF-IDF

TF-IDF: all words treated mostly independently

RNN: understands sequence/order

So:
    dog bites man
    man bites dog

become different meanings.
"""
"""
Big Problem of RNN


Sentence: "The movie I watched yesterday with my friends was not good"

By the time the model reaches: good

it may forget: not

because memory weakens over long sequences.

This becomes: Vanishing Gradient Problem

Simple intuition: old information fades away
like weak human memory.

RNN memory is:

    short-term
    unstable for long sequences
"""