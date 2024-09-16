---
layout: post
title: "Dependencies"
---

<a href="https://colab.research.google.com/github/nishzsche/nishzsche.github.io/blob/gh-pages/draw_graph_dev.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
# Dependencies


import networkx as nx
import matplotlib.pyplot as plt
```


```python
def generate_path_graph(m):
    G = nx.path_graph(m)  # networkx has a built-in path graph generator
    return G

def draw_path_graph(m):
    # Generate the graph
    G = generate_path_graph(m)

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Layout for visualizing the graph
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', font_size=16, font_weight='bold')

    # Show the graph
    plt.title(f"Path Graph with {m} vertices")
    plt.show()
```


```python
# Example

draw_path_graph(10)
```


    
![png](draw_graph_dev_files/draw_graph_dev_3_0.png)
    
