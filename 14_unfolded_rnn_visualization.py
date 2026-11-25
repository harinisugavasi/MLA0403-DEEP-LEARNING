# CODE:
import networkx as nx
import matplotlib.pyplot as plt
G=nx.DiGraph()
steps=4
for t in range(steps):
 G.add_node(f"x{t}",layer="input"); G.add_node(f"h{t}",layer="hidden"); G.add_node(f"y{t}",layer="output")
 G.add_edge(f"x{t}",f"h{t}"); G.add_edge(f"h{t}",f"y{t}")
 if t>0:G.add_edge(f"h{t-1}",f"h{t}")
pos={}
for t in range(steps): pos[f"x{t}"]=(t,0); pos[f"h{t}"]=(t,1); pos[f"y{t}"]=(t,2)
nx.draw(G,pos,with_labels=True,node_size=900,arrows=True); plt.title("RNN Unrolled Across Time")
plt.savefig("14_unfolded_rnn_visualization.png"); plt.show()

# INPUT:
# Input: four recurrent time steps.
#

# OUTPUT:
# Output: directed graph visualization of input, hidden and output states.
