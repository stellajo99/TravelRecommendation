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
vector = tfidf_vectorizer.transform(text).toarray()


# In[17]:


from sklearn.cluster import DBSCAN
import numpy as np

vector = np.array(vector)

# eps는 점으로부터의 반경, min_sample은 eps 내 최소 점의 개수 기준
# min_sample은 한 점으로부터 반경 EPS 인 원을 그렸을 때 그 점이 한 군집의
# 중심이 되기 위해 eps 안에 필요한 최소한의 점 개수
# 너무 작은 수면 잡음이 많고 너무 많은 개수의 군집이 형성될 수 있음
model = DBSCAN(eps=0.5, min_samples=5, metric="cosine")
result=model.fit_predict(vector)


# In[18]:


result


# In[19]:


df['result']=result
df.head()


# In[20]:


count=0
for cluster_num in set(result):
    if(cluster_num==-2):
        continue

    else:
        print("cluster num : {}".format(cluster_num))
        temp_df=df[df['result']==cluster_num]
        for id in temp_df['id']:
            #print(id)
            count+=1
        print(count)
        print()
        count=0
        


# In[8]:


# eps를 0.1부터 1.0까지 높여보며 cluster 개수와 noise point 개수 확인
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score, completeness_score,v_measure_score

for i in range(1,10,1):
    model = DBSCAN(eps=i*0.1, min_samples=10, metric="cosine")
    model.fit(vector)
    labels=model.labels_
    n_clusters_=len(set(labels))-(1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d"%n_clusters_)
    print("Estimated number of noise points: %d"%n_noise_)


# In[7]:


# eps를 0.1부터 1.0까지 높여보며 cluster 개수와 noise point 개수 확인
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score, completeness_score,v_measure_score

for i in range(1,10,1):
    model = DBSCAN(eps=i*0.1, min_samples=5, metric="cosine")
    model.fit(vector)
    labels=model.labels_
    n_clusters_=len(set(labels))-(1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d"%n_clusters_)
    print("Estimated number of noise points: %d"%n_noise_)
    print("Silhouette Coefficient: %0.3f" %metrics.silhouette_score(vector,labels))
    print()


# In[21]:


from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score, completeness_score,v_measure_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

range_eps = [5,6,7]

for i in range(5,6,1):
    model = DBSCAN(eps=i*0.1, min_samples=5, metric="cosine")
    model.fit(vector)
    labels=model.labels_
    n_clusters_=len(set(labels))-(1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d"%n_clusters_)
    print("Estimated number of noise points: %d"%n_noise_)
    print("Silhouette Coefficient: %0.3f" %metrics.silhouette_score(vector,labels))
    print()  
    silhouette_avg = metrics.silhouette_score(vector,labels)
    sample_silhouette_values = silhouette_samples(vector, labels)
    
    y_lower = 10
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(18,7)

    ax1.set_xlim([-0.1,1])
    ax1.set_ylim([0,2782+(n_clusters_+1)*10])
    
    for i in range(n_clusters_):
        
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters_)
        ax1.fill_betweenx(np.arange(y_lower,y_upper),
                         0,
                         ith_cluster_silhouette_values,
                         facecolor=color,
                         edgecolor=color,
                         alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i),fontsize=15)
        y_lower = y_upper + 10
    
    ax1.set_title("The silhouette plot for the various clusters. - DBSCAN",fontsize=20)
    ax1.set_xlabel("The silhouette coefficient values",fontsize=20)
    ax1.set_ylabel("Cluster label",fontsize=20)
    
    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')
    ax1.set_yticks([])
    ax1.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])
    
    
#plt.show()
plt.savefig('dbscan_sil.png')


# In[ ]:





# In[ ]:




