# CODE:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X=StandardScaler().fit_transform(load_iris().data); pca=PCA(n_components=2)
Z=pca.fit_transform(X); Xrec=pca.inverse_transform(Z)
mse=np.mean((X-Xrec)**2); print("Reconstruction MSE (2 components):",mse)
plt.scatter(X[:,0],Xrec[:,0],alpha=.7); plt.title("Original vs reconstructed first feature")
plt.xlabel("Original"); plt.ylabel("Reconstructed")
plt.savefig("20_probabilistic_pca_reconstruction.png"); plt.show()

# INPUT:
# Input: standardized Iris data; retain two principal components as a PPCA-style reconstruction demonstration.
#

# OUTPUT:
# Output: reconstruction MSE and original/reconstructed scatter plot.
