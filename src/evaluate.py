import numpy as np
import tensorflow as tf
from src.preprocess import prepare_data

X, y = prepare_data()
model = tf.keras.models.load_model("models/pos_rnn.keras")
loss, acc = model.evaluate(X, y, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Token Accuracy (includes padding handling by mask where supported): {acc*100:.2f}%")

pred = np.argmax(model.predict(X, verbose=0), axis=-1)
true = np.argmax(y, axis=-1)
mask = X != 0
masked_acc = (pred[mask] == true[mask]).mean()
print(f"Non-padding Token Accuracy: {masked_acc*100:.2f}%")
