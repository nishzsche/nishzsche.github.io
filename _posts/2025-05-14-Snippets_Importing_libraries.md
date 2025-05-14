---
layout: post
title: "Importing a library that is not in Colaboratory"
---

# Importing a library that is not in Colaboratory

To import a library that's not in Colaboratory by default, you can use `!pip install` or `!apt-get install`.


```
!pip install matplotlib-venn
```


```
!apt-get -qq install -y libfluidsynth1
```

# Install 7zip reader [libarchive](https://pypi.python.org/pypi/libarchive)


```
# https://pypi.python.org/pypi/libarchive
!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive
```

# Install GraphViz & [PyDot](https://pypi.python.org/pypi/pydot)


```
# https://pypi.python.org/pypi/pydot
!apt-get -qq install -y graphviz && pip install pydot
import pydot
```

# Install [cartopy](http://scitools.org.uk/cartopy/docs/latest/)


```
!pip install cartopy
import cartopy
```
