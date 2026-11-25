# CODE:
import tensorflow as tf
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train=x_train.astype("float32")/255.; x_test=x_test.astype("float32")/255.
model=tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dense(10,activation="softmax")])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
h=model.fit(x_train,y_train,epochs=3,validation_split=.1,verbose=1)
print("Test metrics:",model.evaluate(x_test,y_test,verbose=0))
plt.plot(h.history["accuracy"],label="train"); plt.plot(h.history["val_accuracy"],label="validation")
plt.legend(); plt.title("MLP Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.savefig("09_mlp_mnist.png"); plt.show()

# INPUT:
# Input: MNIST handwritten-digit dataset downloaded by Keras.
#

# OUTPUT:
# Output: test loss/accuracy and training-validation accuracy graph.
