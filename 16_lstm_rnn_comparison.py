# CODE:
import numpy as np, tensorflow as tf, pandas as pd
import matplotlib.pyplot as plt
seq=np.sin(np.arange(0,100,.5)); win=10
X=np.array([seq[i:i+win] for i in range(len(seq)-win)])[...,None]; y=seq[win:]
results={}
for kind,layer in [("SimpleRNN",tf.keras.layers.SimpleRNN(16)),("LSTM",tf.keras.layers.LSTM(16))]:
 m=tf.keras.Sequential([layer,tf.keras.layers.Dense(1)]); m.compile("adam","mse"); h=m.fit(X,y,epochs=4,verbose=0)
 results[kind]=h.history["loss"][-1]; print(kind,"final MSE",results[kind])
plt.bar(results.keys(),results.values()); plt.title("RNN vs LSTM final training loss"); plt.ylabel("MSE")
plt.savefig("16_lstm_rnn_comparison.png"); plt.show()

# INPUT:
# Input: sliding windows from sine sequence.
#

# OUTPUT:
# Output: final MSE for SimpleRNN and LSTM and comparison plot.
