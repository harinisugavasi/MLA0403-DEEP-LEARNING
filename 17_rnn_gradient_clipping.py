# CODE:
import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
X=tf.constant(np.random.randn(32,15,1),dtype=tf.float32); y=tf.constant(np.random.randn(32,1),dtype=tf.float32)
m=tf.keras.Sequential([tf.keras.layers.SimpleRNN(16,input_shape=(15,1)),tf.keras.layers.Dense(1)])
opt=tf.keras.optimizers.SGD(.01); raw_norms=[]; clipped_norms=[]
for _ in range(30):
 with tf.GradientTape() as tape: loss=tf.reduce_mean((m(X)-y)**2)
 grads=tape.gradient(loss,m.trainable_variables); raw_norms.append(float(tf.linalg.global_norm(grads)))
 clipped,_=tf.clip_by_global_norm(grads,1.0); clipped_norms.append(float(tf.linalg.global_norm(clipped))); opt.apply_gradients(zip(clipped,m.trainable_variables))
print("Initial/final gradient norm:",raw_norms[0],raw_norms[-1])
plt.plot(raw_norms,label="raw"); plt.plot(clipped_norms,label="clipped"); plt.legend(); plt.title("Gradient clipping")
plt.savefig("17_rnn_gradient_clipping.png"); plt.show()

# INPUT:
# Input: random batches of sequence data and regression targets.
#

# OUTPUT:
# Output: gradient norm traces before/after clipping.
