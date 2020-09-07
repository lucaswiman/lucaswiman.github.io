---
title: Notes on how this blog is generated





---


Based on https://www.fast.ai/2020/01/20/nb2md/

Notes:
1. It's very important that the title be in a single cell at the beginning. _only_ the `# My title` part will be included in the processed markdown file.
2. This uses nbdev and the file `make-post.py`. The command to make a post from a jupyter notebook is:
```bash
./make-post.py notebooks/YYYY-MM-DD-post-title.ipynb
```

Here is an image that is copy/pasted into the cell:
![image.png](/images/2020-09-06-example-post_files/att_00000.png)

Here is some LaTeX: $\alpha(x^y) = z$

Here is some code:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
for x in range(10):
    print(x)
```

</div>
<div class="output_area" markdown="1">

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


</div>

</div>

Here is some more markdown.

And here is some code that generates an SVG:

<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

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

</div>
<div class="output_area" markdown="1">


![svg](/images/2020-09-06-example-post_files/output_3_0.svg)


</div>

</div>

Here is some code that generates latex from sympy:
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
import sympy
x, y = sympy.symbols('x y')
sympy.expand((x + y) ** 20)
```

</div>
<div class="output_area" markdown="1">




$\displaystyle x^{20} + 20 x^{19} y + 190 x^{18} y^{2} + 1140 x^{17} y^{3} + 4845 x^{16} y^{4} + 15504 x^{15} y^{5} + 38760 x^{14} y^{6} + 77520 x^{13} y^{7} + 125970 x^{12} y^{8} + 167960 x^{11} y^{9} + 184756 x^{10} y^{10} + 167960 x^{9} y^{11} + 125970 x^{8} y^{12} + 77520 x^{7} y^{13} + 38760 x^{6} y^{14} + 15504 x^{5} y^{15} + 4845 x^{4} y^{16} + 1140 x^{3} y^{17} + 190 x^{2} y^{18} + 20 x y^{19} + y^{20}$



</div>

</div>

And the same thing, but hidden (should be blank):

And the same thing, but with [the _input_ hidden](https://forums.fast.ai/t/fastpages-hide-code-but-show-output/64344) (should have Latex but not the code):
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

</div>
<div class="output_area" markdown="1">




$\displaystyle x^{20} + 20 x^{19} y + 190 x^{18} y^{2} + 1140 x^{17} y^{3} + 4845 x^{16} y^{4} + 15504 x^{15} y^{5} + 38760 x^{14} y^{6} + 77520 x^{13} y^{7} + 125970 x^{12} y^{8} + 167960 x^{11} y^{9} + 184756 x^{10} y^{10} + 167960 x^{9} y^{11} + 125970 x^{8} y^{12} + 77520 x^{7} y^{13} + 38760 x^{6} y^{14} + 15504 x^{5} y^{15} + 4845 x^{4} y^{16} + 1140 x^{3} y^{17} + 190 x^{2} y^{18} + 20 x y^{19} + y^{20}$



</div>

</div>

And the same thing, but with the _output_ hidden (should have the code but not the Latex):
<div class="codecell" markdown="1">
<div class="input_area" markdown="1">

```python
# hide_output
import sympy
x, y = sympy.symbols('x y')
sympy.expand((x + y) ** 20)
```

</div>
<div class="output_area" markdown="1">

</div>

</div>
