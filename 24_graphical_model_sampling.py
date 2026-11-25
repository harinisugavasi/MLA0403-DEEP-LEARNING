# CODE:
import numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt
rng=np.random.default_rng(3); n=1000
cloudy=rng.binomial(1,.5,n); rain=np.where(cloudy,rng.binomial(1,.8,n),rng.binomial(1,.2,n))
sprinkler=np.where(cloudy,rng.binomial(1,.1,n),rng.binomial(1,.5,n))
wet=np.where((rain+sprinkler)>0,rng.binomial(1,.9,n),rng.binomial(1,.01,n))
df=pd.DataFrame({"Cloudy":cloudy,"Rain":rain,"Sprinkler":sprinkler,"WetGrass":wet})
print(df.head()); print("Estimated P(WetGrass=1):",df.WetGrass.mean())
G=nx.DiGraph([("Cloudy","Rain"),("Cloudy","Sprinkler"),("Rain","WetGrass"),("Sprinkler","WetGrass")])
nx.draw(G,with_labels=True,node_size=1500); plt.title("Sampling Bayesian graph")
plt.savefig("24_graphical_model_sampling.png"); plt.show()

# INPUT:
# Input: sampled binary variables under illustrative conditional probabilities.
#

# OUTPUT:
# Output: sample dataframe, estimated wet-grass probability and graph.
