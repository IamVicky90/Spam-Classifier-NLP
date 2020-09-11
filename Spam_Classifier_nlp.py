# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:23:29 2020

@author: Muhammad Waqas
"""


import pandas as pd
import nltk
import re
df=pd.read_csv("SMSSpamCollection",sep="\t",names=["Label","Messages"])
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
corpus=[]
for i in range(len(df)):
    review=re.sub("[^a-zA-Z]"," ",df["Messages"][i])
    review=review.lower()
    review=review.split()
    review=[PorterStemmer().stem(word) for word in review if word not in set(stopwords.words("english"))]
    review=" ".join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()
y=pd.get_dummies(df["Label"],drop_first=True)
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=42)
X_res,y_res=smk.fit_sample(X,y)
from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train=train_test_split(X_res,y_res,random_state=42,test_size=0.2)
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(x_train,y_train)
predict=model.predict(x_test)
from sklearn.metrics import confusion_matrix
cv=confusion_matrix(y_test,predict)
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,predict)
