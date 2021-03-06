# -*- coding: utf-8 -*-


# Authors: Brent McManus, John Karasev, Donovan Ellison, Long Le, Trevor Leake
#
# Description: Makes long/short-term recursive neural network and trains the
#   model to label tweets as having positive or negative sentiments based on
#   vocabulary and word ordering.
#
# Data source:      https://www.kaggle.com/kazanova/sentiment140
# Modelled after:   https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow



# Libraries used
# The only non-standard Python library required is tensorflow.
import numpy as np
import tensorflow as tf
from random import randint
import datetime
import tensorboard
import csv
import re
import json
import sys
from os.path import isfile



# Takes a tweet's text as string. Returns list of words dropping hashtags, mentions and special characters.
def preprocess_tweet(string):
    # First we'll strip special characters out using regex
    strip_special_chars = re.compile("[^A-Za-z0-9#@ ]+")
    string = string.lower().replace("<br />", " ")
    string = re.sub(strip_special_chars, "", string.lower())
    string = string.split()

    # Used to filter out user mentions and hashtags common in tweets.
    mention_and_haghtag_filter = lambda string: string is not None and string[0] != '#' and string[0] != '@'
    return list(filter(filter_func, string))


# Takes filepath to our csv of labelled tweets. Returns list of processed tweets and list of sentiments.
def read_preprocess(file):
    with open(file, encoding = 'latin1') as csvfile:
        tweets = []
        sentiments = []
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        for row in reader:
            tweets.append(preprocess_tweet(row[5]))
            if row[0] == "0":
                sentiments.append(0)
            else:
                sentiments.append(1)
        return tweets, sentiments


# Used for saving our tweets and sentiments after being processed.
def save(tweets,filename1,sentiments,filename2):
    with open(filename1, 'w') as of:
        json.dump(tweets, of)
    with open(filename2, 'w') as of:
        json.dump(sentiments, of)


# Multi-purpose progress bar / equals-symbol breeder / reminder that you're doing something useful (DONT CLOSE THE TERMINAL)
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', suffix))
    sys.stdout.flush()
    if count == total:
        print("")


# Reads globally set parameters: batchSize, maxTweetLength, and ids. Returns array of selected ids and their labels (?) BUG
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxTweetLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,533333)
            labels.append([1,0])
        else:
            num = randint(1066667,1600000)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels


# Reads globally set parameters: batchSize, maxTweetLength, and ids. Returns array of selected ids and their labels (?) BUG
def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxTweetLength])
    for i in range(batchSize):
        num = randint(533334,1066667)
        if (num < 800000):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels


# Program parameters
numTweets = 1600000
maxTweetLength = 35
tweetCounter = 0
batchSize = 100
lstmUnits = 64
numClasses = 2
epochs = 4
iterations = int(((epochs*2*numTweets)/3)/batchSize) #full batch = 44445
numDimensions = 300
ids = np.zeros((numTweets,maxTweetLength), dtype = 'int32')


# If tweets haven't been processed and saved
if not isfile('tweets.json'):
    tweets, sentiments = read_preprocess('training.1600000.processed.noemoticon.csv')
    print("Data processed")
    save(tweets, "tweets.json", sentiments, "sentiments.json")
    print("Data saved to \"./sentiments.json\"")

# If processed tweets are already saved
else:
    with open('tweets.json') as f:
        tweets = json.load(f)
    with open('sentiments.json') as f:
        sentiments = json.load(f)
    print("Loaded data")


# Read in dictionary and associated vectors
wordsList = np.load('wordsList.npy')
wordsList = wordsList.tolist() #convert numpy to list
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('wordVectors.npy')


# Vectorize all of the tweets, takes a very long time to run. However it loads saved data when possible.
if isfile('idsMatrix.npy'):
    ids = np.load('idsMatrix.npy')
else
    for tweet in tweets:
        progress(tweetCounter,numTweets)
        wordCounter = 0
        for word in tweet:
            try:
                ids[tweetCounter][wordCounter] = wordsList.index(word)
            except:
                ids[tweetCounter][wordCounter] = 399999 #vector for unknown words
            wordCounter += 1
            if wordCounter >= maxTweetLength:
                break
        tweetCounter += 1
    np.save('idsMatrix', ids)


with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordVectors,ids[0]).eval().shape)

tf.reset_default_graph()

# Create placeholder tensors
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxTweetLength])

# Create tensor for ....BUG
data = tf.Variable(tf.zeros([batchSize, maxTweetLength, numDimensions]), dtype=tf.float32)

# Grab vectors associated with our input data. Similar to the concept of word embeddings.
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

#tensorboard code
tf.summary.scalar('Loss',loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# Uncomment to load model instead of training anew
#sess = tf.InteractiveSession()
#saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint('models'))

#training on tweets
for i in range(iterations):
    progress(i,iterations)
    #next batch of tweets
    nextBatch, nextBatchLabels = getTrainBatch()
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    #write summary to tensorboard every 50 batches
    if i % 50 == 0:
        summary = sess.run(merged, {input_data: nextBatch,
                                    labels: nextBatchLabels})
        writer.add_summary(summary,i)

    #save the network every 44,445 training iterations size of the training set
    if i % (iterations - 1) == 0 and i != 0:
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt",
                               global_step = i)
        print("Saved to %s" % save_path)
writer.close()

iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch()
    print("Accuracy for this batch:", (sess.run(accuracy, {
            input_data: nextBatch, labels: nextBatchLabels}))*100)
