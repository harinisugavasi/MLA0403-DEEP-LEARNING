# CODE:
import numpy as np
from scipy.special import entr, rel_entr
import matplotlib.pyplot as plt
p = np.array([.7, .2, .1]); q = np.array([.5, .3, .2])
entropy = np.sum(entr(p)); cross_entropy = -np.sum(p*np.log(q)); kl = np.sum(rel_entr(p,q))
print("Entropy:", entropy); print("Cross-entropy:", cross_entropy); print("KL divergence:", kl)
plt.bar(["Entropy","Cross-Entropy","KL"], [entropy,cross_entropy,kl])
plt.title("Information Measures"); plt.ylabel("Value")
plt.savefig("04_entropy_cross_entropy_kl.png"); plt.show()

# INPUT:
# Input: probability vectors p=[0.7,0.2,0.1], q=[0.5,0.3,0.2].
#

# OUTPUT:
# Output: entropy, cross-entropy, KL divergence and bar chart.
