# CODE:
import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
x=tf.constant([[1.],[2.],[3.],[4.]]); y=tf.constant([[2.],[4.],[6.],[8.]])
optimizers={"GD":tf.keras.optimizers.SGD(.05),"Momentum":tf.keras.optimizers.SGD(.05,momentum=.9),"Adam":tf.keras.optimizers.Adam(.05)}
hist={}
for name,opt in optimizers.items():
 w=tf.Variable(0.); losses=[]
 for epoch in range(80):
  with tf.GradientTape() as tape:
   loss=tf.reduce_mean((w*x-y)**2)
  grad=tape.gradient(loss,[w]); opt.apply_gradients(zip(grad,[w])); losses.append(float(loss))
 hist[name]=losses; print(name,"final weight",float(w),"final loss",losses[-1])
for name,losses in hist.items(): plt.plot(losses,label=name)
plt.legend(); plt.title("Optimizer Loss Comparison"); plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.savefig("06_optimizers_comparison.png"); plt.show()

# INPUT:
# Input: x=[1,2,3,4], y=2x, learning rate 0.05.
#

# OUTPUT:
# Output: optimizer final weights/losses and loss curves.
