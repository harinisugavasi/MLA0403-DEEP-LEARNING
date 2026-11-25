# CODE:
import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(5); v0=np.array([[1.,0.,1.,0.]])
W=rng.normal(0,.2,(4,3)); vb=np.zeros(4); hb=np.zeros(3); sig=lambda z:1/(1+np.exp(-z))
def sample(p): return (rng.random(p.shape)<p).astype(float)
v=v0.copy(); energies=[]
for k in range(30):
 hp=sig(v@W+hb); h=sample(hp); vp=sig(h@W.T+vb); v=sample(vp)
 energies.append(float(-np.sum(v@W*h)))
 print(f"step {k+1}: visible sample {v.astype(int)}")
plt.plot(energies); plt.title("Gibbs chain energy trace"); plt.xlabel("Step"); plt.ylabel("Approx energy")
plt.savefig("26_gibbs_contrastive_divergence.png"); plt.show()

# INPUT:
# Input: initial binary visible state; randomly initialized small RBM.
#

# OUTPUT:
# Output: Gibbs samples and approximate energy trace (illustrative CD chain).
