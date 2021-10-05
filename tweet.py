# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 22:57:58 2021

@author: Administrator
"""

import os
os.getcwd()
os.chdir("C:/Users/Administrator/Desktop/PYTHON/NLP")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tweet=pd.read_csv("Tweets.csv")
nrow,ncol=tweet.shape
print(f"there are {nrow} rows and {ncol} columns in this dataset")

tweet.describe().T
tweet.info()
tweet.isnull().any()
sns.heatmap(tweet.isnull(),cmap='plasma')
sns.pairplot(hue='airline',data=tweet)
tweet['negativereason'].value_counts()
tweet['airline_sentiment'].value_counts()
tweet['negativereason'].apply(type)
tweet['negativereason']=tweet['negativereason'].fillna('No review')
sns.countplot(data=tweet,y='negativereason')
sns.countplot(data=tweet,x='airline_sentiment',hue='airline')
sns.countplot(data=tweet,x='airline',hue='airline_sentiment')

tweet['negativereason_confidence'].apply(type)
tweet['negativereason_confidence']=tweet['negativereason_confidence'].fillna(tweet['negativereason_confidence'].mean())
tweet['airline_sentiment_gold'].value_counts()

tweet.columns
tweet['text'].apply(len).idxmax()
tweet['length']=tweet['text'].apply(len)
tweet['length'].plot.hist()
tweet['length'].describe()
tweet['length'].max()
tweet[tweet['length']==186]['text'].iloc[0]
tweet[tweet.airline_sentiment=='positive']['length']
tweet[tweet.length==136]['text'].iloc[0]
tweet[tweet.airline_sentiment=='neutral']['length'].value_counts()
tweet[tweet.length==140]['text'].iloc[0]

pd.DataFrame(tweet[['airline_sentiment','tweet_location','airline']])

data=tweet[['text','airline_sentiment']]

x=data['text']
y=data['airline_sentiment']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2,random_state=100)

#TO CONVERT A COLLECTION OF RAW DOCUMENTS TO A MATRIX OF TF-IDF FEATURES
from sklearn.feature_extraction.text import TfidfVectorizer
tfid=TfidfVectorizer(stop_words='english')
tfid.fit(xtrain)

xtrain_tf=tfid.transform(xtrain)
xtest_tf=tfid.transform(xtest)

xtrain_tf

from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression(max_iter = 1000)
lr_model.fit(xtrain_tf,ytrain)

y_pred=lr_model.predict(xtest_tf)

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
print(classification_report(ytest,y_pred))
cm=confusion_matrix(ytest,y_pred)


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(xtrain_tf,ytrain)

y_pred2=nb.predict(xtest_tf)
print(classification_report(ytest,y_pred2))
cm2=confusion_matrix(ytest,y_pred2)

plot_confusion_matrix(lr_model,xtest_tf,ytest)

plot_confusion_matrix(nb,xtest_tf,ytest)

lr_model.predict([['ok flight']])

from sklearn.pipeline import Pipeline
pipe=Pipeline([('tfid',TfidfVectorizer()),('lr_model',LogisticRegression())])
pipe.fit(x,y)

pipe.predict(['Ok flight'])
