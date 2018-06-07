import csv
import re
import json
import itertools
from word2vec import Word2Vec
import sys
import numpy as np

vocab = None


def filter_func(string):
    if string is None or string[0] == '@' or string[0] == '#':
        return False
    else:
        return True


def preprocess_tweet(string):
    strip_special_chars = re.compile("[^A-Za-z0-9#@ ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    string = string.split()
    return list(filter(filter_func, string))


def preprocess(self):
    self.tweets = []  # holds text
    self.sentiment = []  # holds label
    with open(self.file,
              encoding='latin1') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            self.tweets.append(preprocess_tweet(row[5]))
            if row[0] == "0" or row[0] == "1":
                self.sentiment.append(0)
            elif row[0] == "2":
                self.sentiment.append(1)
            else:
                self.sentiment.append(2)


def export_vocab(tweets, vocab_size, export=True):
    words = []
    for tweet in tweets:
        words.extend(tweet)
    vocab = Word2Vec.vocab_to_num(words, vocab_size)
    if export:
        np.save('./data/vocab.npy', vocab)
    return vocab


def map_func(x):
    if x not in vocab.keys():
        return 0
    else:
        return vocab[x]

    # prints the progress of a process
    # Vladimir Ignatyev  https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console


def progress(count, total, suffix=''):
    bar_len = 60

    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', suffix))
    sys.stdout.flush()
    if count == total:
        print("")


def map_sentences(tweets, vocabs, export=True, name="mapped_tweets.npy"):
    global vocab  # global for map function
    if vocab is None:
        try:
            vocab = np.load("./data/vocab.npy").item()
        except FileNotFoundError:
            print("please create a vocabulary table first")
            exit(1)
    else:
        vocab = vocabs
    for index in range(len(tweets)):
        progress(index, len(tweets), "mapping words to integers")
        tweets[index] = list(map(map_func, tweets[index]))
    print(str(tweets))
    if export:
        print("writing mapped tweets to " + name)
        np.save("./data/" + name, tweets)
    return tweets


def create_dataset(tweets, window, datafile="mapped_tweets.npy", export=True):
    if tweets is None:
        try:
            tweets = np.load(datafile).item()
        except FileNotFoundError:
            print("cannot find " + datafile)
            exit(1)
    contexts, neighbors = Word2Vec.create_dataset(tweets, window)
    if export:
        print("saving train set to file")
        contexts = np.array(contexts)
        neighbors = np.array(neighbors)
        contexts.tofile('./data/npcontexts.dat')
        neighbors.tofile('./data/npneighbors.dat')
