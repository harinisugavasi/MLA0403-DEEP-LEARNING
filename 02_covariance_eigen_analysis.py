# CODE:
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
X = np.array([[2, 1], [3, 2], [4, 4], [5, 3], [6, 5]], dtype=float)
df = pd.DataFrame(X, columns=["feature1", "feature2"])
cov = np.cov(X.T); vals, vecs = np.linalg.eig(cov)
print("Data:\n", df); print("Covariance matrix:\n", cov)
print("Eigenvalues:", vals); print("Eigenvectors:\n", vecs)
plt.scatter(X[:,0], X[:,1]); plt.title("Feature data"); plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.savefig("02_covariance_eigen_analysis.png"); plt.show()

# INPUT:
# Input: a small two-feature dataset.
#

# OUTPUT:
# Output: covariance matrix, eigenvalues/eigenvectors, and scatter plot.
