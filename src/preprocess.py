import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

DATA_PATH = "data/pos_data.txt"

def load_tagged_data():
    sentences, tags = [], []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            words, labels = [], []
            for item in line.strip().split():
                if "/" in item:
                    word, tag = item.rsplit("/", 1)
                    words.append(word.lower())
                    labels.append(tag)
            if words:
                sentences.append(words)
                tags.append(labels)
    return sentences, tags

def prepare_data():
    sentences, tags = load_tagged_data()
    words = sorted(set(w for s in sentences for w in s))
    tagset = sorted(set(t for ts in tags for t in ts))

    word2idx = {"<PAD>":0, "<OOV>":1}
    for w in words:
        word2idx[w] = len(word2idx)

    tag2idx = {"<PAD>":0}
    for t in tagset:
        tag2idx[t] = len(tag2idx)
    idx2tag = {v:k for k,v in tag2idx.items()}

    max_len = max(len(s) for s in sentences)
    X = [[word2idx.get(w,1) for w in s] for s in sentences]
    y = [[tag2idx[t] for t in ts] for ts in tags]

    X = pad_sequences(X, maxlen=max_len, padding="post", value=0)
    y = pad_sequences(y, maxlen=max_len, padding="post", value=0)
    y_cat = to_categorical(y, num_classes=len(tag2idx))

    os.makedirs("models", exist_ok=True)
    with open("models/config.pkl","wb") as f:
        pickle.dump({
            "word2idx":word2idx, "tag2idx":tag2idx, "idx2tag":idx2tag,
            "max_len":max_len, "vocab_size":len(word2idx), "num_tags":len(tag2idx)
        }, f)
    return X, y_cat

if __name__ == "__main__":
    X, y = prepare_data()
    print("Input:", X.shape, "Target:", y.shape)
