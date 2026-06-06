import re
s="hello123"
result=re.sub(r"\d","",s) 
#\d mean sigle digit , but here due to sub function substitution funtion 
#But re.sub() applies it repeatedly across the whole string, so all matching digits get removed.
print(result)
text="<tag>Hello</tag><b>Hi</b>"
result1=re.sub(r"<.*?>", "", text)
"""
| Part | Meaning             |
| ---- | ------------------- |
| `<`  | literal `<`         |
| `.`  | any character       |
| `*`  | zero or more times  |
| `?`  | non-greedy matching |
Without ?, regex becomes greedy.
Greedy matching might capture:

<br>hello<div>

ENTIRE thing at once.

But non-greedy *? stops at first >.

So it correctly matches:

<br>
<div>
"""
print(result1)