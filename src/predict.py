import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_resources():
    model = tf.keras.models.load_model("models/pos_rnn.keras")
    with open("models/config.pkl","rb") as f:
        c = pickle.load(f)
    return model, c

def tag_sentence(sentence):
    model, c = load_resources()
    words = sentence.lower().split()
    seq = [c["word2idx"].get(w, c["word2idx"]["<OOV>"]) for w in words]
    x = pad_sequences([seq], maxlen=c["max_len"], padding="post", truncating="post")
    pred = np.argmax(model.predict(x, verbose=0)[0], axis=-1)
    result = []
    for i, word in enumerate(words[:c["max_len"]]):
        result.append((word, c["idx2tag"].get(int(pred[i]), "UNKNOWN")))
    return result

if __name__ == "__main__":
    sentence = input("Enter sentence: ")
    for word, tag in tag_sentence(sentence):
        print(f"{word}: {tag}")
