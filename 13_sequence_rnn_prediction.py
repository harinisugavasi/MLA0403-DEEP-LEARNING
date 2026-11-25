# CODE:
import numpy as np, pandas as pd, tensorflow as tf
import matplotlib.pyplot as plt
series=np.sin(np.arange(0,100,.5)); df=pd.DataFrame({"value":series})
window=10; X=np.array([series[i:i+window] for i in range(len(series)-window)]); y=series[window:]
X=X[...,None]; model=tf.keras.Sequential([tf.keras.layers.SimpleRNN(16,input_shape=(window,1)),tf.keras.layers.Dense(1)])
model.compile("adam","mse"); model.fit(X,y,epochs=5,verbose=0)
pred=model.predict(X,verbose=0).ravel(); print(df.head()); print("Example prediction:",pred[0],"target:",y[0])
plt.plot(y,label="target"); plt.plot(pred,label="prediction"); plt.legend(); plt.title("RNN sequence prediction")
plt.savefig("13_sequence_rnn_prediction.png"); plt.show()

# INPUT:
# Input: generated sine series converted to sliding windows of length 10.
#

# OUTPUT:
# Output: sample prediction and target-versus-prediction graph.
