import string 
import re

def clean_text(text):

    #To lower case
    text=text.lower()

    #remove HTML tags
    text=re.sub(r"<.*?>","",text)

    #remove punctuation
    text=text.translate(str.maketrans("","",string.punctuation))
    """
    maketrans() → creates mapping/instructions
    translate() → applies those instructions
    str.maketrans(x, y, z)
    | Parameter | Meaning                |
    | --------- | ---------------------- |
    | `x`       | characters to replace  |
    | `y`       | replacement characters |
    | `z`       | characters to DELETE   |

    example:
    table = str.maketrans("abc", "xyz")
    | Original | New |
    | -------- | --- |
    | a        | x   |
    | b        | y   |
    | c        | z   |

    "text abc".translate(table)
    output is "text xyz"

    """

    return text
print(clean_text("Awesome Movie!!! <br /><br /> I LOVED it.")) 

