# CODE:
import numpy as np
import matplotlib.pyplot as plt
X=np.array([[0,0],[0,1],[1,0],[1,1]]); w=np.array([1.,1.]); b=-.5
z=X@w+b; yhat=(z>=0).astype(int)
print("Weighted sums:",z); print("Predictions:",yhat)
plt.scatter(X[:,0],X[:,1],c=yhat); plt.title("Perceptron predictions"); plt.xlabel("x1"); plt.ylabel("x2")
plt.savefig("07_perceptron_forward.png"); plt.show()

# INPUT:
# Input: four binary feature pairs; fixed weights [1,1], bias -0.5.
#

# OUTPUT:
# Output: weighted sums, threshold predictions and scatter plot.
