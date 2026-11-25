# CODE:
import tensorflow as tf
import matplotlib.pyplot as plt
(x,y),(xt,yt)=tf.keras.datasets.mnist.load_data(); x=x[...,None]/255.; xt=xt[...,None]/255.
results={}
for filters in [16,32]:
 m=tf.keras.Sequential([tf.keras.layers.Conv2D(filters,3,activation="relu",input_shape=(28,28,1)),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Flatten(),tf.keras.layers.Dense(10,activation="softmax")])
 m.compile("adam","sparse_categorical_crossentropy",metrics=["accuracy"]); m.fit(x,y,epochs=1,verbose=0)
 results[filters]=m.evaluate(xt,yt,verbose=0)[1]; print("Filters",filters,"test accuracy",results[filters])
plt.bar([str(k) for k in results],list(results.values())); plt.title("CNN filter comparison"); plt.xlabel("Filters"); plt.ylabel("Test accuracy")
plt.savefig("12_cnn_hyperparameter_comparison.png"); plt.show()

# INPUT:
# Input: MNIST; compare convolution filter counts 16 and 32.
#

# OUTPUT:
# Output: test accuracy for each configuration and bar chart.
