# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:10:39 2019

@author: Lohith Kumar Reddy
"""


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from numpy.linalg import norm as nm
import random as rn
import sys

import scipy.spatial as sp




def read_csv_files(file_name):
    
    df = pd.read_csv(file_name)
    k = []
    prewords = []
    for index,rows in df.iterrows():
        k = df.iloc[[index],2]
        k = pd.Series.to_string(k)
        prewords.append(word_tokenize(k))

    return prewords




def tfidf(docs):
    tfidf_vectorizer=TfidfVectorizer(use_idf=True)
    fitted_vectorizer=tfidf_vectorizer.fit(docs)
    tfidf_vectorizer_vectors=fitted_vectorizer.transform(docs)
    feature_names = tfidf_vectorizer.get_feature_names()
    df = pd.DataFrame(tfidf_vectorizer_vectors.toarray().transpose(),index=tfidf_vectorizer.get_feature_names())
    return df

def count_values(docs):
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(docs)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix,columns=count_vectorizer.get_feature_names())
    return df
    
def preprocessing(words):
    # words = [" ".join(x) for x in words]
    table = str.maketrans('', '', string.punctuation)
    
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't" ,'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
    
    from nltk.stem.snowball import SnowballStemmer
    englishstemmer = SnowballStemmer("english")
    tokens  = []
    st = []
    for word in words:        
        tokens.append(st)
        l = []
        m = []
        n = []        
        st = []
        for j in word:
            l = j.lower()            
            m = l.translate(table)            
            if m.isalpha() :
                n = m            
            if tuple(n) not in (stop_words):
                n = str(n)
                
                
                st.append(englishstemmer.stem(n))
                      
    tokens = [" ".join(x) for x in tokens]   
    
    return tokens

def column_sum(df):
    sum_column = df.sum(axis=0)
    return sum_column

def cosine_sim(a,b): 
    b = b.transpose()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1 - (dot_product / (norm_a * norm_b))

def norm_for_cosine(a,b):
    dot_prod = []
    
    for i in range(a.shape[0]):
        dot_pro = np.dot(a[i],b[i])
        norm_i = np.linalg.norm(a[i])
        norm_j = np.linalg.norm(b[i])
        dot_prod.append(1 - (dot_pro / (norm_i * norm_j)))
    dot_prod = np.array(dot_prod)           
    return dot_prod

def euclidean(a,b):
    
    return np.linalg.norm(a-b)

def findcolminmax(items):
    n = len(items[0])
    minima = [sys.maxsize for i in range(n)]
    maxima = [-sys.maxsize for i in range(n)]
    for item in items:
        for f in range(len(item)):
            if(item[f] < minima[f]):
                minima[f]=item[f]
            if(item[f] > maxima[f]):
                maxima[f]=item[f]
    return minima,maxima

def initializemeans(items,k):
    f = len(items[0])
    cmin,cmax = findcolminmax(items)
    means = [[0 for i in range(f)] for j in range(k)]
    for mean in means:
        for i in range(len(mean)):
            mean[i] = rn.uniform(cmin[i]+1,cmax[i]-1)
    return means

def kmeans(k,dataset,distance,epsilon=0):
    history_centroids = []
    
    #Distance Type
    if distance == "euclidean":
        dist_method = euclidean
    if distance == "cosine":
        dist_method = cosine_sim
        
    
    
    #Getting the shape of the data
    num_instances,num_features = dataset.shape
    
    # Defining the initial Centroids
    prototypes = initializemeans(dataset,k)
    
    #Putting the centroids in history
    history_centroids.append(prototypes)
    
    #Variable to track of centroids at each iteration
    prototypes_old = np.zeros(np.shape(prototypes))
    
    # To store clusters
    belongs_to = np.zeros((num_instances,1)) # This is for each Data Point
    prototypes = np.asarray(prototypes, dtype=np.float)
    Norm = norm_for_cosine(prototypes,prototypes_old)
    
    
    iteration = 0
    norm_history = []
    iterations = []
    
    # While loop where we assign the centroids to the point
    
    while Norm.all() > epsilon:
    
        iteration += 1
        
        print(iteration)
        # Finding the distance between the old centroids and new centroids
        Norm = norm_for_cosine(prototypes,prototypes_old)
        prototypes_old = prototypes
        
        # For each distance in the dataset
        for index_instance,instance in enumerate(dataset):
            dist_vect = np.zeros((k,1))
            # Distance between point and centroid
            for index_prototype,prototype in enumerate(prototypes):
                prototype = prototype.reshape(1, -1)
                instance = instance.reshape(1, -1)
                dist_vect[index_prototype]= dist_method(prototype,instance)
            belongs_to[index_instance] = np.argmin(dist_vect)
        tmp_prototype = np.zeros((k,num_features))
        
        # Now for each initial cluster and the cluster after that, The optimisation
        for index in range(len(prototypes)):
            instances_close = []
            for i in range(len(belongs_to)):
                if belongs_to[i] == index:
                    instances_close.append(i)
            prototype = np.mean(dataset[instances_close],axis = 0)
            tmp_prototype[index,:] = prototype
        prototypes = tmp_prototype
        
        prototypes = np.nan_to_num(prototypes)
        history_centroids.append(prototype)
              
        norm_history.append(Norm)
        
        iterations.append(iteration)
    
    return prototypes,history_centroids,belongs_to 
#####calculating the number of clusers based on data, K value is changeable
    
def no_clusters_Sil(data):    
    
    sil = []
    kmax = 20
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 
    for k in range(2, kmax+1):
        KMEANS = KMeans(n_clusters = k).fit(data)
        labels = KMEANS.labels_
        sil.append(silhouette_score(data, labels, metric = 'euclidean'))
           
    print(sil.index(max(sil)))
    return sil.index(max(sil))  

   
  

presentences = read_csv_files('book1.csv')
docs = preprocessing(presentences)
# uniquenessvaluesvalues = tfidf(docs)
df = count_values(docs)
sum_of_columns = column_sum(df)
df = df.append(sum_of_columns,ignore_index=True)


mean_of_values = sum_of_columns.mean()
df_transpose = df.transpose()
rslt_df = df_transpose.loc[df_transpose[len(df_transpose.columns)-1] > mean_of_values*2]
rslt_df = rslt_df.iloc[:, :-1]
df = rslt_df.transpose()
dataset = df.to_numpy()
k = no_clusters_Sil(dataset)
centroids, history_centroids,belongs_to= kmeans(k,dataset,'cosine')
#print("centroid 1",centroids)


