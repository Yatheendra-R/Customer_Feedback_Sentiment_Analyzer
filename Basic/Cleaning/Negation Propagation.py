negation_words = {
    "not",
    "no",
    "never"
}

tokens = [
    "this",
    "movie",
    "is",
    "not",
    "good"
]

new_tokens = []

negate = False

for word in tokens:

    # CHECK NEGATION WORD

    if word in negation_words:

        negate = True

        # keep original negation word
        new_tokens.append(word)

        continue

    # APPLY NEGATION

    if negate:

        new_tokens.append("NOT_" + word)

        # reset negate flag
        negate = False

    # NORMAL WORD

    else:

        new_tokens.append(word)


# OUTPUT


print(new_tokens)