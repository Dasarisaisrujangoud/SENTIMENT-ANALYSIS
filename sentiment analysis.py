#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names


# In[11]:


data_pth = r'C:\Users\saisr\Downloads\data.csv'
df = pd.read_csv(data_pth)
df.head()


# In[15]:


print(df.shape)


# In[16]:


df['Sentiment'].value_counts().plot(kind='bar')


# In[18]:


df['Sentiment'].value_counts().plot(kind='hist')


# In[21]:


df['Sentiment'].value_counts().plot(kind='pie')


# In[29]:


def word_character(words):
    return dict([(word, True) for word in words])


# In[3]:


positive_word = [ 'awesome', 'interesting','fabulous','lovely','outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_word = ['dislike', 'horrible','gross','bad','worse', 'terrible','useless', 'hate', 'meaningless','waste',':(' ]
neutral_word = [ 'movie','think','game','program','project','the','sound','was','is','actors','did','know','words','not' ]


# In[4]:


positive_character = [(word_character(pos), 'pos') for pos in positive_word]
negative_character = [(word_character(neg), 'neg') for neg in negative_word]
neutral_character = [(word_character(neu), 'neu') for neu in neutral_word]
 
train_set = negative_character + positive_character + neutral_character


# In[5]:


classifier = NaiveBayesClassifier.train(train_set)


# In[6]:


neg = 0
pos = 0
sentence = "awesome match, I've enjoyed  it"
sentence = sentence.lower()
words = sentence.split(' ')
for word in words:
    classResult = classifier.classify( word_character(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1
 
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))


# In[ ]:




