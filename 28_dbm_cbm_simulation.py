# CODE:
import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(8); visible=6; hidden1=4; hidden2=3
W1=rng.normal(0,.1,(visible,hidden1)); W2=rng.normal(0,.1,(hidden1,hidden2))
v=np.array([1.,0.,1.,0.,1.,0.]); sig=lambda z:1/(1+np.exp(-z))
h1=sig(v@W1); h2=sig(h1@W2); vrec=sig(h1@W1.T)
print("Layer 1 activation:",h1); print("Layer 2 activation:",h2); print("Visible reconstruction:",vrec)
plt.bar(range(visible),vrec); plt.title("Deep Boltzmann-style reconstruction"); plt.xlabel("Visible unit"); plt.ylabel("Activation")
plt.savefig("28_dbm_cbm_simulation.png"); plt.show()

# INPUT:
# Input: six-unit binary visible vector and small random two-layer weights.
#

# OUTPUT:
# Output: layer activations and reconstruction plot; illustrative DBM-style forward simulation.
