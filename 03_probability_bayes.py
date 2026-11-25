# CODE:
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt
x = np.linspace(-4, 4, 200); density = norm.pdf(x, loc=0, scale=1)
print("Binomial P(X=3), n=10,p=.5:", binom.pmf(3, 10, .5))
# Bayes: disease prevalence .01, sensitivity .90, false positive rate .05
prior, sensitivity, false_positive = .01, .90, .05
posterior = sensitivity*prior/(sensitivity*prior + false_positive*(1-prior))
print("P(disease | positive):", posterior)
plt.plot(x, density); plt.title("Standard Normal Distribution"); plt.xlabel("x"); plt.ylabel("Density")
plt.savefig("03_probability_bayes.png"); plt.show()

# INPUT:
# Input: standard normal parameters; binomial n=10,p=0.5; Bayes prior/sensitivity/false-positive rates.
#

# OUTPUT:
# Output: selected distribution probability, Bayes posterior, distribution plot.
