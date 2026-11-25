# CODE:
import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
tf.random.set_seed(2)
x=tf.constant(np.random.randn(20,1,1),dtype=tf.float32); target=tf.constant([[0.]],dtype=tf.float32)
rnn=tf.keras.layers.SimpleRNN(4); dense=tf.keras.layers.Dense(1)
with tf.GradientTape() as tape:
 h=rnn(x); pred=dense(h); loss=tf.reduce_mean((pred-target)**2)
grads=tape.gradient(loss,rnn.trainable_variables+dense.trainable_variables)
norms=[float(tf.norm(g)) for g in grads if g is not None]
print("Loss:",float(loss)); print("Gradient norms:",norms)
plt.bar(range(len(norms)),norms); plt.title("BPTT gradient norms"); plt.xlabel("Parameter group"); plt.ylabel("Norm")
plt.savefig("15_bptt_gradient_analysis.png"); plt.show()

# INPUT:
# Input: random sequence of 20 steps and scalar target.
#

# OUTPUT:
# Output: loss, parameter gradient norms and bar graph.
