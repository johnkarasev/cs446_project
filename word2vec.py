import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tflearn
import collections
import json
import os
import sys
import ijson


class Word2Vec(object):

    def __init__(self, x, y, vocab_size, embedd_size, learning_rate, nce_sample_size, save, skipgram=True,
                 batch_size=32):
        self.savefile = save
        self.vocab_size = vocab_size
        self.input = x
        self.batch_size = batch_size
        self.target = y
        self.optimizer, self.loss, self.x, self.y, self.sess, self.embed = Word2Vec.create_nn(vocab_size, embedd_size,
                                                                                              learning_rate,
                                                                                              nce_sample_size,
                                                                                              batch_size, skipgram)

    @staticmethod
    def progress(count, total, suffix=''):
        bar_len = 60

        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', suffix))
        sys.stdout.flush()
        if count == total:
            print("")

    @staticmethod
    def vocab_to_num(words, vocab_size):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(vocab_size - 1))
        word_to_int = {}
        for word, _ in count:
            word_to_int[word] = len(word_to_int)
        with open("vocab.json", "w") as f:
            json.dump(word_to_int, f, indent=2)
        return word_to_int

    @staticmethod
    def create_dataset(sentences, window):
        neighbor_words = []
        context_words = []
        for sentence in range(sentences.shape[0]):
            Word2Vec.progress(sentence, sentences.shape[0], "building dataset")
            contexts = sentences[sentence][window:-window]
            for index in range(len(contexts)):
                context = contexts[index]
                neighbors = np.array([])
                prev_words = sentences[sentence][index: window + index]
                next_words = sentences[sentence][index + window + 1:2 * window + index + 1]
                neighbors = np.append(neighbors, [prev_words, next_words]).flatten().tolist()
                for i in range(window * 2):
                    context_words.append(context)
                    neighbor_words.append(neighbors[i])
        return context_words, neighbor_words

    @staticmethod
    def create_nn(vocab_size, embedding_size, learning_rate, nce_sample_size, batch_size, skipgram=True):
        if skipgram:
            x = tf.placeholder(tf.int32, shape=[batch_size], name="contexts")
            y = tf.placeholder(tf.int32, shape=[batch_size, 1], name="neighbors")
        else:
            x = tf.placeholder(tf.int32, shape=[None, ], name="neighbors")
            y = tf.placeholder(tf.int32, shape=[None, ], name="contexts")
        Embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="word_embeddings")
        nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=tf.sqrt(1 / embedding_size)),
                                  name="nce_weights")
        nce_biases = tf.Variable(tf.zeros([vocab_size]), name="nce_biases")
        word_embed = tf.nn.embedding_lookup(Embedding, x, name="word_embed_lookup")
        # train_labels = tf.reshape(y, [tf.shape(y)[0], 1])
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=y,
                                             inputs=word_embed,
                                             num_sampled=nce_sample_size,
                                             num_classes=vocab_size,
                                             num_true=1))
        optimizer = tf.contrib.layers.optimize_loss(loss,
                                                    tf.train.get_global_step(),
                                                    learning_rate,
                                                    "Adam",
                                                    clip_gradients=5.0,
                                                    name="optimizer")
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return optimizer, loss, x, y, sess, Embedding

    def train(self, epochs):
        with open("vocab.json") as fp:
            vocab = json.load(fp)
        # x_train, x_test, y_train, y_test = train_test_split(self.input, self.target)
        print(type(self.input))
        num_batches = self.input.shape[0] // self.batch_size
        print(num_batches)
        saver = tf.train.Saver([self.embed])
        for epoch in range(epochs):
            for i in range(num_batches):
                if i != range(num_batches - 1):
                    x_batch = self.input[i * self.batch_size:i * self.batch_size + self.batch_size]
                    y_batch = self.target[i * self.batch_size:i * self.batch_size + self.batch_size]
                else:
                    x_batch = self.input[i * self.batch_size:]
                    y_batch = self.target[i * self.batch_size:]

                _, l = self.sess.run([self.optimizer, self.loss],
                                     feed_dict={self.x: x_batch.reshape((x_batch.shape[0])),
                                                self.y: y_batch.reshape((y_batch.shape[0], 1))})
                if i % 100 == 0:
                    print("STEP " + str(i) + " of " + str(num_batches) + " LOSS: " + str(l))
                if l < 5.0 and i > 100000:
                    embed = self.embed[:].eval(session=self.sess)
                    embed.tofile("vectorspace" + str(embed.shape[0]) + "x" + str(embed.shape[1]) + ".np")
                    return
        embed = self.embed[:].eval(session=self.sess)
        embed.tofile("vectorspace" + str(embed.shape[0]) + "x" + str(embed.shape[1]) + ".np")
        save_path = saver.save(self.sess, os.path.join("tf_log", self.savefile))