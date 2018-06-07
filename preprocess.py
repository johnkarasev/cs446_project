import csv
import re
import json


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