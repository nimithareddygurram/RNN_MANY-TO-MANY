import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, TimeDistributed
from src.preprocess import prepare_data

tf.random.set_seed(42)
X, y = prepare_data()
with open("models/config.pkl","rb") as f:
    c = pickle.load(f)

model = Sequential([
    Embedding(c["vocab_size"], 64, mask_zero=True),
    SimpleRNN(64, return_sequences=True, activation="tanh"),
    TimeDistributed(Dense(c["num_tags"], activation="softmax"))
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, validation_split=0.2, epochs=100, batch_size=2, verbose=1)
model.save("models/pos_rnn.keras")
print("Saved: models/pos_rnn.keras")
