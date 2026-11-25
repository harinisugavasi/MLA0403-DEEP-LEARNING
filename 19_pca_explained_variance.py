# CODE:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X=StandardScaler().fit_transform(load_iris().data); pca=PCA().fit(X)
print("Explained variance ratio:",pca.explained_variance_ratio_)
print("Cumulative ratio:",np.cumsum(pca.explained_variance_ratio_))
plt.plot(range(1,5),np.cumsum(pca.explained_variance_ratio_),marker="o"); plt.title("PCA cumulative explained variance")
plt.xlabel("Components"); plt.ylabel("Cumulative variance")
plt.savefig("19_pca_explained_variance.png"); plt.show()

# INPUT:
# Input: standardized Iris feature dataset.
#

# OUTPUT:
# Output: component variance ratios and cumulative variance graph.
