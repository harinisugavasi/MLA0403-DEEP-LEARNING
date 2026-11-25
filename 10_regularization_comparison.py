# CODE:
import tensorflow as tf
import matplotlib.pyplot as plt
(x,y),_=tf.keras.datasets.mnist.load_data(); x=x.astype("float32")/255.
def build(reg):
 layers=[tf.keras.layers.Flatten(input_shape=(28,28))]
 if reg=="L2": layers.append(tf.keras.layers.Dense(128,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(.001)))
 elif reg=="Dropout": layers += [tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dropout(.4)]
 else: layers.append(tf.keras.layers.Dense(128,activation="relu"))
 layers.append(tf.keras.layers.Dense(10,activation="softmax"))
 m=tf.keras.Sequential(layers); m.compile("adam","sparse_categorical_crossentropy",metrics=["accuracy"]); return m
histories={}
for kind in ["L2","Dropout","EarlyStopping"]:
 m=build(kind); callbacks=[tf.keras.callbacks.EarlyStopping(patience=1,restore_best_weights=True)] if kind=="EarlyStopping" else []
 histories[kind]=m.fit(x,y,epochs=3,validation_split=.1,callbacks=callbacks,verbose=0).history["val_accuracy"]
 print(kind,"validation accuracy:",histories[kind][-1])
for k,v in histories.items(): plt.plot(v,label=k)
plt.legend(); plt.title("Regularization validation accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.savefig("10_regularization_comparison.png"); plt.show()

# INPUT:
# Input: MNIST training images; three regularization variants.
#

# OUTPUT:
# Output: validation accuracy comparison and graph.
