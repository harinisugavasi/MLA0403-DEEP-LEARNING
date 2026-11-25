# CODE:
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
data=load_iris(); X=StandardScaler().fit_transform(data.data)
fa=FactorAnalysis(n_components=2,random_state=0).fit(X); Z=fa.transform(X)
print("Factor loadings:\n",pd.DataFrame(fa.components_,columns=data.feature_names))
plt.scatter(Z[:,0],Z[:,1],c=data.target); plt.title("Latent factor scores"); plt.xlabel("Factor 1"); plt.ylabel("Factor 2")
plt.savefig("21_factor_analysis.png"); plt.show()

# INPUT:
# Input: standardized Iris dataset, two latent factors.
#

# OUTPUT:
# Output: factor loading table and latent score visualization.
