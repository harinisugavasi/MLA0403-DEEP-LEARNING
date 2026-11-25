# CODE:
import numpy as np
import matplotlib.pyplot as plt
a = np.array([1, 2, 3]); b = np.array([4, 5, 6])
A = np.array([[1, 2], [3, 4]]); B = np.array([[5, 6], [7, 8]])
print("Vector addition:", a + b)
print("Dot product:", np.dot(a, b))
print("Matrix addition:\n", A + B)
print("Matrix multiplication:\n", A @ B)
print("Transpose of A:\n", A.T)
plt.plot(a, label="Vector a"); plt.plot(b, label="Vector b")
plt.title("Vector values"); plt.xlabel("Index"); plt.ylabel("Value"); plt.legend()
plt.savefig("01_vector_matrix_operations.png"); plt.show()

# INPUT:
# Input: vectors a=[1,2,3], b=[4,5,6]; matrices A=[[1,2],[3,4]], B=[[5,6],[7,8]].
#

# OUTPUT:
# Output: vector/matrix arithmetic and a saved/displayed vector plot.
