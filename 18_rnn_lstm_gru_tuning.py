# CODE:
import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
# Compact manual hyperparameter sweep; KerasTuner may be used for extended tuning.
seq=np.sin(np.arange(0,80,.5)); w=8
X=np.array([seq[i:i+w] for i in range(len(seq)-w)])[...,None]; y=seq[w:]
scores={}
for units in [8,16]:
 for kind,layer_cls in [("RNN",tf.keras.layers.SimpleRNN),("LSTM",tf.keras.layers.LSTM),("GRU",tf.keras.layers.GRU)]:
  m=tf.keras.Sequential([layer_cls(units,input_shape=(w,1)),tf.keras.layers.Dense(1)])
  m.compile("adam","mse"); hist=m.fit(X,y,epochs=2,verbose=0)
  scores[f"{kind}-{units}"]=hist.history["loss"][-1]; print(f"{kind}, units={units}: loss={scores[f'{kind}-{units}']:.4f}")
plt.bar(scores.keys(),scores.values()); plt.xticks(rotation=45); plt.title("RNN family/unit sweep"); plt.ylabel("Final MSE")
plt.tight_layout(); plt.savefig("18_rnn_lstm_gru_tuning.png"); plt.show()

# INPUT:
# Input: sine sequence; compare RNN/LSTM/GRU with 8 and 16 units.
#

# OUTPUT:
# Output: losses across configurations and bar chart.
