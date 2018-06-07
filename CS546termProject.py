# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:51:26 2018

@author: brent
"""
import numpy as np
import tensorflow as tf
#import matplotlib.pylot as plt
import csv
import re
import json
import sys

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
    return list(filter(filter_func,string))

def read_preprocess(file):
    with open(file, encoding = 'latin1') as csvfile:
        tweets = []
        sentiments = []
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        for row in reader:
            tweets.append(preprocess_tweet(row[5]))
            if row[0] == "0" or row[0] == "1":
                sentiments.append(0)
            elif row[0] == "2":
                sentiments.append(1)
            else:
                sentiments.append(2)
        return tweets, sentiments

def save(tweets,filename1,sentiments,filename2):
    with open(filename1, 'w') as of:
        json.dump(tweets, of)
    with open(filename2, 'w') as of:
        json.dump(tweets, of)
  
#neat functionality but very slow progress bar      
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', suffix))
    sys.stdout.flush()
    if count == total:
        print("")
  
numTweets = 1600000
maxTweetLength = 35
tweetCounter = 0
ids = np.zeros((numTweets,maxTweetLength), dtype = 'int32')
#intial reading and preprocessing
#tweets,sentiments = read_preprocess('training.1600000.processed.noemoticon.csv')
#save(tweets,"tweets.json",sentiments,"sentiments.json")
with open('tweets.json') as f:
    tweets = json.load(f)
with open('sentiments.json') as f:
    sentiments = json.load(f)
    
#read in dictionary and associated vectors 
wordsList = np.load('wordslist.npy')
wordsList = wordsList.tolist() #convert numpy to list
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('wordVectors.npy')

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
    
            