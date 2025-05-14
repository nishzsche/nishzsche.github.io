---
layout: post
title: "Section 1"
---

<a href="https://colab.research.google.com/github/nishzsche/nishzsche.github.io/blob/gh-pages/Markdown_Guide.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## What is Markdown?

Colab has two types of cells: text and code. Text cells are formatted using a simple markup language called Markdown.


To see the Markdown source, double-click a text cell, showing both the Markdown source and the rendered version. Above the Markdown source there is a toolbar to assist editing.

## Reference

Markdown | Preview
--- | ---
`**bold text**` | **bold text**
`*italicized text*` or `_italicized text_` | *italicized text*
`` `Monospace` `` | `Monospace`
`~~strikethrough~~` | ~~strikethrough~~
`[A link](https://www.google.com)` | [A link](https://www.google.com)
`![An image](https://www.google.com/images/rss.png)` | ![An image](https://www.google.com/images/rss.png)


---
Headings are rendered as titles.

```markdown
# Section 1
# Section 2
## Sub-section under Section 2
### Sub-section under the sub-section under Section 2
# Section 3
```

# Section 1
# Section 2
## Sub-section under Section 2
### Sub-section under the sub-section under Section 2
# Section 3

The table of contents, available on the left side of Colab, is populated using at most one section title from each text cell.

---

```markdown
>One level of indentation
```

>One level of indentation


```markdown
>>Two levels of indentation
```

>>Two levels of indentation

---

Code blocks

````
```python
print("a")
```
````

```python
print("a")
```

---

Ordered lists:
```markdown
1. One
1. Two
1. Three
```
1. One
1. Two
1. Three

---

Unordered lists:
```markdown
* One
* Two
* Three
```
* One
* Two
* Three

---

Equations:

```markdown
$y=x^2$

$e^{i\pi} + 1 = 0$

$e^x=\sum_{i=0}^\infty rac{1}{i!}x^i$

$rac{n!}{k!(n-k)!} = {n 