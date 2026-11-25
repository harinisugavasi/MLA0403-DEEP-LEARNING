# CODE:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
rng=np.random.default_rng(7); X=np.linspace(0,1,35).reshape(-1,1); y=np.sin(2*np.pi*X[:,0])+rng.normal(0,.15,35)
xx=np.linspace(0,1,300).reshape(-1,1)
plt.scatter(X,y,label="Samples")
for degree in [1,4,15]:
 model=make_pipeline(PolynomialFeatures(degree),LinearRegression()).fit(X,y)
 plt.plot(xx,model.predict(xx),label=f"degree {degree}")
 print(f"Degree {degree} training MSE:",mean_squared_error(y,model.predict(X)))
plt.legend(); plt.title("Underfit / Fit / Overfit illustration"); plt.xlabel("x"); plt.ylabel("y")
plt.savefig("05_bias_variance_analysis.png"); plt.show()

# INPUT:
# Input: synthetic noisy sinusoidal observations.
#

# OUTPUT:
# Output: training MSE by polynomial degree and model-fit plot.
