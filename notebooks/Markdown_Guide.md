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

$e^x=\sum_{i=0}^\infty \frac{1}{i!}x^i$

$\frac{n!}{k!(n-k)!} = {n \choose k}$

$A_{m,n} =
 \begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n}
 \end{pmatrix}$
```

$y=x^2$

$e^{i\pi} + 1 = 0$

$e^x=\sum_{i=0}^\infty \frac{1}{i!}x^i$

$\frac{n!}{k!(n-k)!} = {n \choose k}$

$A_{m,n} =
 \begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n}
 \end{pmatrix}$


 ---

Tables:
```markdown
First column name  | Second column name
-------------------|------------------
Row 1, Col 1       | Row 1, Col 2
Row 2, Col 1       | Row 2, Col 2

```

First column name  | Second column name
-------------------|------------------
Row 1, Col 1       | Row 1, Col 2
Row 2, Col 1       | Row 2, Col 2

---

Horizontal rules:
```markdown
---
```

---




## Differences between Colab Markdown and other Markdown dialects

Colab uses [marked.js](https://github.com/chjj/marked) and so is similar but not quite identical to the Markdown used by Jupyter and GitHub.

Colab supports (MathJax) $\LaTeX$ equations like Jupyter, but does not allow HTML tags in the Markdown. Colab does not support some GitHub additions like emojis and to-do checkboxes.

If HTML must be included in a Colab notebook, see the [%%html magic](/notebooks/basic_features_overview.ipynb#scrollTo=qM4myQGfQboQ).

## Useful references

* [GitHub Markdown basics](https://help.github.com/articles/markdown-basics/)
* [GitHub flavored Markdown](https://help.github.com/articles/github-flavored-markdown/)
* [Original Markdown spec: Syntax](http://daringfireball.net/projects/markdown/syntax)
* [Original Markdown spec: Basics](http://daringfireball.net/projects/markdown/basics)
* [marked.js library used by Colab](https://github.com/chjj/marked)
* [LaTex mathematics for equations](https://en.wikibooks.org/wiki/LaTeX/Mathematics)
