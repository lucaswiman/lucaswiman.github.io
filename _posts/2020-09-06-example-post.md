# Posting to github pages using jupyter notebooks



```python
for x in range(10):
    print(x)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9


Here is some more mardown.

And here is some code that generates an SVG:


```python
import pygraphviz as pgv
from IPython.display import Image, SVG, display
import networkx as nx

def draw(graph):
    svg = nx.nx_agraph.to_agraph(graph).draw(prog='dot',format='svg')
    display(SVG(svg))

    
import networkx as nx
G = nx.DiGraph()
G.add_edge('a', 'b')
G.add_edge('a', 'c')
G.add_edge('b', 'd')
G.add_edge('c', 'd')
G.add_edge('e', 'f')
G.add_edge('h', 'f')
G.add_edge('confounder', 'h')
G.add_edge('confounder', 'f')

draw(G)
```


![svg](2020-09-06-example-post_files/output_2_0.svg)


Here is some code that generates latex from sympy:

```python
import sympy
x, y = sympy.symbols('x y')
sympy.expand((x + y) ** 20)
```




$\displaystyle x^{20} + 20 x^{19} y + 190 x^{18} y^{2} + 1140 x^{17} y^{3} + 4845 x^{16} y^{4} + 15504 x^{15} y^{5} + 38760 x^{14} y^{6} + 77520 x^{13} y^{7} + 125970 x^{12} y^{8} + 167960 x^{11} y^{9} + 184756 x^{10} y^{10} + 167960 x^{9} y^{11} + 125970 x^{8} y^{12} + 77520 x^{7} y^{13} + 38760 x^{6} y^{14} + 15504 x^{5} y^{15} + 4845 x^{4} y^{16} + 1140 x^{3} y^{17} + 190 x^{2} y^{18} + 20 x y^{19} + y^{20}$



And the same thing, but hidden:
