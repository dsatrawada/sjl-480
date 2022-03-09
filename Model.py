#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Imports
import pandas as pd
import numpy as np
from collections import Counter
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import ttest_ind


# In[56]:


# Get the data into a dataframe to manupulate and do preprocessing
dataFile = 'AmItheAsshole_with_comments_10.csv'
dataFile2 = 'AmItheAsshole_hot.csv'
df = pd.read_csv(dataFile)
df2 = pd.read_csv(dataFile2)
df = df[['comments', 'body']]
df2 = df2[['comments', 'body']]
df = df.append(df2, ignore_index=True)

gt_array = []
np.array(gt_array)
count = 0
LIMIT = 300
for row in df.itertuples():
    arr = row[1].lower()
    nta_count = arr.count('nta') + arr.count('nah')
    yta_count = arr.count('yta') + arr.count('esh')
    if nta_count == 0 and yta_count == 0:
        ratio = .5
    elif nta_count > 0:
        ratio = yta_count / nta_count
    else:
        ratio = 1
    
    if ratio > .45 and ratio < .55:
        gt_array.append(2)
    elif nta_count > yta_count and count < LIMIT:
        gt_array.append(0)
        count += 1
    elif yta_count > nta_count:
        gt_array.append(1)
    else:
        df = df.drop(row[0], axis=0)

gt_data = {'label': gt_array}
gt_df = pd.DataFrame(gt_data)
# print(gt_df['label'].value_counts())


bag_of_words = (
    df['body'].
    str.lower().                  # convert all letters to lowercase
    str.replace("[^\w\s]", " ").  # replace non-alphanumeric characters by whitespace
    str.split()                   # split on whitespace
)

raw_frequency = bag_of_words.apply(Counter)

df['selftext'] = raw_frequency

tf = pd.DataFrame(list(raw_frequency),index=raw_frequency.index)
columns = list(tf.columns)
tf = tf.fillna(0)

stopFile = 'stopwords.txt'
f = open(stopFile, "r")
stop_words = []
for line in f:
    words = line.split(',')
for word in words:
    word = word.replace('"', "").strip(" ").lower()
    stop_words.append(word)
    
for col in columns:
    if col in stop_words:
        tf = tf.drop([col], axis=1)

# Get document frequencies 
# (How many documents does each word appear in?)
df = (tf > 0).sum(axis=0)

# Get IDFs
idf = np.log(len(tf) / df)
idf.sort_values()

# Calculate TF-IDFs
tf_idf = tf * idf
tf_idf


def preprocess(text):
    
    bag_of_words = (
    text.
    lower().                  # convert all letters to lowercase
    replace("[^\w\s]", " ").  # replace non-alphanumeric characters by whitespace
    split()                   # split on whitespace
    )
    
    for word in bag_of_words:
        if word in stop_words:
            bag_of_words.remove(word)
    
    raw_frequency = Counter(bag_of_words)

    tf2 = tf.append(raw_frequency, ignore_index=True)
    tf2 = tf2.fillna(0)
    
    df = (tf2 > 0).sum(axis=0)

    # Get IDFs
    idf = np.log(len(tf2) / df)
    idf.sort_values()

    # Calculate TF-IDFs
    tf_idf = tf2 * idf
    
    return tf_idf.iloc[-1]
        


# scikit learns version of RandomForests
def randomForest(df, gt_df):
    #implement the random forest algorithm on the data
    X = df
    y = gt_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(X_train, y_train)
    return rf

    # y_pred = rf.predict(X_test)
    # display(y_pred)
    # print('Accuracy:', accuracy_score(y_test, y_pred))
    


# In[87]:


# scikit learns version of RandomForests
def KNN(df, gt_df):
    #implement the random forest algorithm on the data
    X = df
    y = gt_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    return neigh
#     y_pred = neigh.predict(X_test)
#     print('Accuracy:', accuracy_score(y_test, y_pred))


# In[91]:


rf = randomForest(tf_idf, gt_df)


# In[90]:


knn = KNN(tf_idf, gt_df)


# In[ ]:




