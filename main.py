import numpy as np
import json
import preprocess
from word2vec import Word2Vec


def main():
    contexts = np.fromfile("./data/npcontexts.dat", dtype=int)
    neighbors = np.fromfile("./data/npneighbors.dat", dtype=int)
    skipgram = Word2Vec(contexts, neighbors, 35000, 10, 0.001, 64, "sg.ckpt", batch_size=500)
    skipgram.train(2)


if __name__ == "__main__":
    main()
