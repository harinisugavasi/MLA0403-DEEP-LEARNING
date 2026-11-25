# CODE:
import tensorflow as tf
import matplotlib.pyplot as plt
(x,y),_=tf.keras.datasets.mnist.load_data(); x=x[:5000].astype("float32")/255.
# Stacked dense layers serve as a compact DBN-inspired representation learner.
encoder=tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(256,activation="relu"),tf.keras.layers.Dense(64,activation="relu")])
decoder=tf.keras.Sequential([tf.keras.layers.Dense(256,activation="relu",input_shape=(64,)),tf.keras.layers.Dense(784,activation="sigmoid")])
inp=tf.keras.Input((28,28)); z=encoder(inp); out=tf.keras.layers.Reshape((28,28))(decoder(z))
auto=tf.keras.Model(inp,out); auto.compile("adam","binary_crossentropy"); h=auto.fit(x,x,epochs=3,validation_split=.1,verbose=0)
print("Final reconstruction loss:",h.history["loss"][-1])
plt.plot(h.history["loss"],label="train"); plt.plot(h.history["val_loss"],label="validation"); plt.legend(); plt.title("Representation reconstruction loss")
plt.savefig("27_deep_belief_network.png"); plt.show()

# INPUT:
# Input: subset of MNIST images; DBN-inspired stacked representation autoencoder demonstration.
#

# OUTPUT:
# Output: reconstruction loss and loss graph. This compact demo is not layer-wise RBM pretraining.
