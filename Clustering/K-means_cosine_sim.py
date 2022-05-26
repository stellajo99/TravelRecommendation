#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_pickle("dbscan_seq.pkl")


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer

text = [" ".join(noun) for noun in df['short_seq']]

tfidf_vectorizer = TfidfVectorizer(min_df = 10, ngram_range=(1,1))
# min_df는 단어의 최소 빈도값인데, 이때 df는 단어의 수가 아니라 특정 단어가 나타나는 문서의 수
tfidf_vectorizer.fit(text)
vector = tfidf_vectorizer.transform(text)


# In[3]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=13, max_iter=100000)
cluster_label=kmeans.fit_predict(vector)

df['cluster_label']=cluster_label
print(df.sort_values(by=['cluster_label']))


# In[24]:


count=0
for cluster_num in set(cluster_label):
    if(cluster_num==-2):
        continue

    else:
        print("cluster num : {}".format(cluster_num))
        temp_df=df[df['cluster_label']==cluster_num]
        for id in temp_df['id']:
            #print(id)
            count+=1
        print(count)
        print()
        count=0
        


# In[4]:


from sklearn.metrics.pairwise import cosine_similarity

idx=df[df['cluster_label']==1].index
print("cluster 1인 문서들의 인덱스:\n",idx)
print()

comparison_doc = df.iloc[idx[0]]['id']
print("## 유사도 비교 기준 문서 이름:", comparison_doc,' ##')
print()

similarity = cosine_similarity(vector[idx[0]],vector[idx])
print(similarity)


# In[ ]:


https://techblog-history-younghunjo1.tistory.com/114


# In[13]:


from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score, completeness_score,v_measure_score

for i in range(1,5,1):
    kmeans = KMeans(n_clusters=i*2, max_iter=100000)
    kmeans.fit_predict(vector)
    labels=kmeans.labels_
    n_clusters_=len(set(labels))-(1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d"%n_clusters_)
    print("Silhouette Coefficient: %0.3f" %metrics.silhouette_score(vector,labels))
    print()


# In[28]:


from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

range_n_clusters = [2,3,4,5,6,13]

for n_clusters in range_n_clusters:
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(18,7)

    ax1.set_xlim([-0.1,1])
    ax1.set_ylim([0,2782+(n_clusters+1)*10])
    
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100000)
    kmeans.fit_predict(vector)
    labels=kmeans.labels_
    
    silhouette_avg = silhouette_score(vector,labels)
    print("For n_clusters = ",n_clusters, "The average silhouette_score is : ", silhouette_avg)
    
    sample_silhouette_values = silhouette_samples(vector, labels)
    
    y_lower = 10
    
    for i in range(n_clusters):
        
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower,y_upper),
                         0,
                         ith_cluster_silhouette_values,
                         facecolor=color,
                         edgecolor=color,
                         alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')
    ax1.set_yticks([])
    ax1.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])
    
    


# In[11]:


from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

range_n_clusters = [13]

for n_clusters in range_n_clusters:
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(18,7)

    ax1.set_xlim([-0.1,1])
    ax1.set_ylim([0,2782+(n_clusters+1)*10])
    
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100000)
    kmeans.fit_predict(vector)
    labels=kmeans.labels_
    
    silhouette_avg = silhouette_score(vector,labels)
    print("For n_clusters = ",n_clusters, "The average silhouette_score is : ", silhouette_avg)
    
    sample_silhouette_values = silhouette_samples(vector, labels)
    y_lower = 10
    
    for i in range(n_clusters):
        
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        y_upper = y_lower + size_cluster_i
        print(y_upper)
        print()
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower,y_upper),
                         0,
                         ith_cluster_silhouette_values,
                         facecolor=color,
                         edgecolor=color,
                         alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i,str(i),fontsize=15.0)
        y_lower = y_upper + 10
    
    ax1.set_title("The silhouette plot for the various clusters. - K-Means",fontsize=20)
    ax1.set_xlabel("The silhouette coefficient values",fontsize=20)
    ax1.set_ylabel("Cluster label",fontsize=20)
    
    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')
    ax1.set_yticks([])
    ax1.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])
   
plt.savefig('kmeans_sil.png')


# In[30]:


get_ipython().system('pip install yellowbrick')


# In[32]:


from sklearn.cluster import KMeans

from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.datasets import load_nfl

# Load a clustering dataset
X, y = load_nfl()

# Specify the features to use for clustering

# Instantiate the clustering model and visualizer
kmeans = KMeans(n_clusters=13, max_iter=100000)
kmeans.fit_predict(vector)
labels=kmeans.labels_
visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')

visualizer.fit(vector)        # Fit the data to the visualizer
visualizer.show()   

