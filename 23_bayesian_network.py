# CODE:
import networkx as nx
import matplotlib.pyplot as plt
# Graph topology: Cloudy -> Rain, Cloudy -> Sprinkler, Rain/Sprinkler -> WetGrass
G=nx.DiGraph([("Cloudy","Rain"),("Cloudy","Sprinkler"),("Rain","WetGrass"),("Sprinkler","WetGrass")])
pos={"Cloudy":(0,1),"Rain":(-1,0),"Sprinkler":(1,0),"WetGrass":(0,-1)}
nx.draw(G,pos,with_labels=True,node_size=1800,arrows=True); plt.title("Bayesian Network Structure")
plt.savefig("23_bayesian_network.png"); plt.show()
print("Nodes:",list(G.nodes)); print("Edges:",list(G.edges))

# INPUT:
# Input: illustrative directed dependencies among Cloudy, Rain, Sprinkler, WetGrass.
#

# OUTPUT:
# Output: node/edge listing and network diagram. pgmpy can be used to add CPDs/inference.
