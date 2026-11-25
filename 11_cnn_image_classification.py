# CODE:
import tensorflow as tf
import matplotlib.pyplot as plt
(x,y),(xt,yt)=tf.keras.datasets.mnist.load_data()
x=x[...,None].astype("float32")/255.; xt=xt[...,None].astype("float32")/255.
m=tf.keras.Sequential([tf.keras.layers.Conv2D(32,3,activation="relu",input_shape=(28,28,1)),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Flatten(),tf.keras.layers.Dense(64,activation="relu"),tf.keras.layers.Dense(10,activation="softmax")])
m.compile("adam","sparse_categorical_crossentropy",metrics=["accuracy"]); h=m.fit(x,y,epochs=2,validation_split=.1,verbose=1)
print("Test evaluation:",m.evaluate(xt,yt,verbose=0))
plt.plot(h.history["val_accuracy"]); plt.title("CNN validation accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.savefig("11_cnn_image_classification.png"); plt.show()

# INPUT:
# Input: MNIST images reshaped to 28x28x1.
#

# OUTPUT:
# Output: test metrics and validation accuracy graph.
