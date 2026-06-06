"""
LSTM (Long Short-Term Memory)

Goal:

help the model remember important information for long periods.

Main Idea of LSTM

Instead of simple memory like RNN:

small unstable memory

LSTM introduces:

controlled memory system

using gates.

Think of it like: a smart memory manager

that decides:

   what to remember
   what to forget
   what to output


Three Main Gates

Understand the intuition.

Forget Gate
   Decides:what old information should be removed

Input Gate
   Decides:what new information should be stored

Output Gate
   Decides: what information to use now

Simple Flow
old memory
   ↓
forget useless info
   ↓
add important new info
   ↓
produce output

Why LSTM Was Huge

It solved:
   long-term dependencies  
   better sentiment understanding
   sequence learning stability

This made deep NLP much stronger before transformers arrived.

Then Another Problem Appeared

Even LSTM struggles when:
   sequences become very long
   important words are very far away
"""