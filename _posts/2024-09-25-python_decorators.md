---
layout: post
title: "Decorators in python"
---

# Decorators in python

Example 1: Treating functions as objects


```python
def shout(text):
  return text.upper()

print(shout("hello"))
```

    HELLO



```python
yell = shout
print(yell("hellO"))
```

    HELLO


Example 2: Passing the function as an argument


```python
def shout(text):
  return text.upper()

def whisper(text):
  return text.lower()

def greet(func):
  greeting = func("Hello there fellow human!")
  print(greeting)

greet(shout)
greet(whisper)
```

    HELLO THERE FELLOW HUMAN!
    hello there fellow human!


Example 3: Returning functions from another function


```python
def create_adder(x):
  def adder(y):
    return x+y
  return adder

# This step returns a function as a result of the call to the outer function
add_10 = create_adder(10)
print(add_10(15))
```

    25
