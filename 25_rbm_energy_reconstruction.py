# CODE:
import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(0); V=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,1.,0.,0.]])
W=rng.normal(0,.1,(4,2)); vb=np.zeros(4); hb=np.zeros(2)
sig=lambda z:1/(1+np.exp(-z))
hprob=sig(V@W+hb); vrec=sig(hprob@W.T+vb); energy=-(V@vb).sum(axis=1)-(hprob@hb).sum(axis=1)-np.sum((V@W)*hprob,axis=1)
print("Hidden probabilities:\n",hprob); print("Reconstruction:\n",vrec); print("Energy:",energy)
plt.imshow(vrec,aspect="auto"); plt.title("RBM reconstruction probabilities"); plt.xlabel("Visible unit"); plt.ylabel("Sample")
plt.savefig("25_rbm_energy_reconstruction.png"); plt.show()

# INPUT:
# Input: three binary visible vectors and initialized RBM weights.
#

# OUTPUT:
# Output: hidden probabilities, reconstruction probabilities, approximate energy and heatmap.
