# CODE:
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1); X=np.array([[0,0],[0,1],[1,0],[1,1]],float); y=np.array([[0],[1],[1],[0]],float)
sig=lambda z:1/(1+np.exp(-z))
W1=np.random.randn(2,4); b1=np.zeros((1,4)); W2=np.random.randn(4,1); b2=np.zeros((1,1)); losses=[]
for epoch in range(5000):
 h=sig(X@W1+b1); out=sig(h@W2+b2); loss=np.mean((out-y)**2); losses.append(loss)
 d2=(out-y)*out*(1-out)*2/len(X); dW2=h.T@d2; db2=d2.sum(0,keepdims=True)
 d1=(d2@W2.T)*h*(1-h); dW1=X.T@d1; db1=d1.sum(0,keepdims=True)
 W2-=.8*dW2; b2-=.8*db2; W1-=.8*dW1; b1-=.8*db1
print("Predictions:\n",out.round(3)); print("Final loss:",loss)
plt.plot(losses); plt.title("Backpropagation loss"); plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.savefig("08_two_layer_backprop.png"); plt.show()

# INPUT:
# Input: XOR dataset (four two-dimensional binary examples).
#

# OUTPUT:
# Output: learned XOR predictions, final MSE and training-loss graph.
