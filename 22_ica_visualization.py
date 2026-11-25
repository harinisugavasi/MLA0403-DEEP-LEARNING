# CODE:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
rng=np.random.default_rng(4); t=np.linspace(0,8,1000)
S=np.c_[np.sin(2*t),np.sign(np.sin(3*t))]; S+=.05*rng.normal(size=S.shape)
A=np.array([[1,2],[2,1]]); mixed=S@A.T
ica=FastICA(n_components=2,random_state=0,whiten="unit-variance",max_iter=1000); recovered=ica.fit_transform(mixed)
fig,ax=plt.subplots(3,1,figsize=(8,6)); ax[0].plot(S); ax[0].set_title("Source signals"); ax[1].plot(mixed); ax[1].set_title("Mixed signals"); ax[2].plot(recovered); ax[2].set_title("ICA components")
plt.tight_layout(); plt.savefig("22_ica_visualization.png"); plt.show()

# INPUT:
# Input: two synthetic independent source signals mixed linearly.
#

# OUTPUT:
# Output: source, mixed, and recovered components plots.
