# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:01:00 2019

@author: Lohith Kumar Reddy

"""
import nltk
import string
import re
import numpy as np
import pandas as pd
import sklearn
from nltk.tokenize import word_tokenize


'''
^^^^^^^^^^^^^ This Block is for text Files^^^^^^^^^^^^^^^^
def split_words(file_name):    
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    # split into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    # remove all tokens that are not alphabetic
    words = [word for word in tokens if word.isalpha()]
    return words
'''



def read_csv_files(file_name):
    
    df = pd.read_csv(file_name)
    k = []
    prewords = []
    for index,rows in df.iterrows():
        k = df.iloc[[index],2]
        k = pd.Series.to_string(k)
        prewords.append(word_tokenize(k))

    return prewords
''' 
    saved_column = df.iloc[:,2]
    print(type(saved_column))
    saved_column = pd.Series.to_string(saved_column)     
'''
      
def preprocessing(words):
    # words = [" ".join(x) for x in words]
       
    tokens  = []
    tok = []
    for i,word in enumerate(words):        
        
        tokens.append(tok)
        tok = []
        for j,tk in enumerate(word):
            tok.append(tk.lower())
    
            
            
    
    
    #Deleting Punctuations
    table = str.maketrans('', '', string.punctuation)
    stripped = []
    tik = []
    for o,k in enumerate(words):        
        
        stripped.append(tik)
        tik = []
        for p,l in enumerate(k):
            tik.append(l.translate(table))
    
      
    # remove all tokens that are not alphabetic
    
    words = []
    wor = []
    for f in stripped:
        
        words.append(wor)
        wor = []
        for g in f:
            if g.isalpha is True:
                wor.append(g)
    
    
    # Removing Stop words
    
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't" ,'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
    
    stoppp = []
    fb = []
    for a,m in enumerate(words):        
        
        stoppp.append(fb)
        fb = []
        for b,l in enumerate(m):
            if l not in (stop_words):
                fb.append(l)
    
    
      
    # Stemming 
    from nltk.stem.snowball import SnowballStemmer
    englishstemmer = SnowballStemmer("english")
    stem_sentence = []
    st = []
    for c in stoppp:
        
        stem_sentence.append(st)
        st = []
        for k in c:            
            st.append(englishstemmer.stem(k))    
 
              
    stem_sentence = [" ".join(x) for x in stem_sentence]
    print(stem_sentence)
    return stem_sentence
  
prewords = read_csv_files('testfile.csv')

final_words = preprocessing(prewords)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvectorizer = TfidfVectorizer(use_idf = True)
tfidfvectorizer_vector = tfidfvectorizer.fit_transform(final_words)




