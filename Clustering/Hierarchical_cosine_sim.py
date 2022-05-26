#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from matplotlib import pyplot as plt


# In[3]:


import nltk
from nltk.tokenize import word_tokenize


# In[2]:


import numpy as np

def cos_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def make_matrix(feats, list_data):
    freq_list = []
    for feat in feats:
        freq=0
        for word in list_data:
            if feat==word:
                freq+=1
        freq_list.append(freq)
    return freq_list

text1='Ta0001 Ta0001 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0002 Ta0003'
text2='Bs0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Kx0001 Kx0001 Ux0001'
text3='Ux0002 Ux0002 Ux0002 Ux0002 Ux0002 Ux0002 Ux0002 Ux0002 Rx0001 Rx0002 Rx0002 Rx0002 Rx0002 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Lx0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001 Ux0001'

v1 = word_tokenize(text2)
v2 = word_tokenize(text3)

v3 = v1+v2
feats = set(v3)

v1_arr = np.array(make_matrix(feats,v1))
v2_arr = np.array(make_matrix(feats,v2))

cs1 = cos_sim(v1_arr, v2_arr)

print('v1<->v2 = ',cs1)


# In[15]:


import numpy as np
import pandas as pd
from tqdm import trange, notebook

df=pd.read_csv('cos_sim_seq.csv')
arr=df['short_seq']

matrix = np.zeros((2782,2782))

def cos_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def make_matrix(feats, list_data):
    freq_list = []
    for feat in feats:
        freq=0
        for word in list_data:
            if feat==word:
                freq+=1
        freq_list.append(freq)
    return freq_list

def cos_sim_matrix(seq1,seq2):
    
    v1 = word_tokenize(seq1)
    v2 = word_tokenize(seq2)
    
    v3 = v1+v2
    feats = set(v3)
    
    v1_arr = np.array(make_matrix(feats,v1))
    v2_arr = np.array(make_matrix(feats,v2))
    
    cs = cos_sim(v1_arr,v2_arr)
    if cs>0.9999999:
        cs=1.0
    distance = 1-cs
    return distance

a = []
for i in notebook.tqdm(range(len(arr))):
    result=[]
    for j in notebook.tqdm(range(len(arr))):
        # print(levenshtein(arr[i],arr[j]))
        result.append(cos_sim_matrix(arr[i],arr[j]))
    #print(result)
    a.append(result)
    
print(a)
    
    


# In[16]:


np.savetxt("hie_cos_dis.txt", a, fmt='%f', delimiter=',')


# In[10]:


import numpy as np
a = np.loadtxt("hie_cos_dis.txt", delimiter=',')


# In[11]:


from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange, notebook

df=pd.read_csv('cos_sim_seq.csv')
arr=df['short_seq']
data = a
arr = df['id']
arr = np.array(arr)
labels = arr

z = linkage(data, 'ward')

print(cut_tree(z, n_clusters=5))

fig = plt.figure(figsize=(25,10))
dn = dendrogram(z, orientation='top', labels=labels)
plt.show()


# In[12]:


cluster_info = cut_tree(z, 13).flatten()

for i in range(len(cluster_info)):
    if(cluster_info[i] == 0):
        cluster_info[i] = 1
    elif(cluster_info[i] == 1):
        cluster_info[i] = 2
    elif(cluster_info[i] == 2):
        cluster_info[i] = 3
    elif(cluster_info[i] == 3):
        cluster_info[i] = 4
    elif(cluster_info[i] == 4):
        cluster_info[i] = 5
    elif(cluster_info[i] == 5):
        cluster_info[i] = 6
    elif(cluster_info[i] == 6):
        cluster_info[i] = 7
    elif(cluster_info[i] == 7):
        cluster_info[i] = 8
    elif(cluster_info[i] == 8):
        cluster_info[i] = 9
    elif(cluster_info[i] == 9):
        cluster_info[i] = 10
    elif(cluster_info[i] == 10):
        cluster_info[i] = 11
    elif(cluster_info[i] == 11):
        cluster_info[i] = 12
    elif(cluster_info[i] == 12):
        cluster_info[i] = 13


df['cluster'] = cluster_info
print(cluster_info)


# In[16]:


from scipy.cluster.hierarchy import fcluster
T = fcluster(z, 3, criterion='maxclust')
print(T)


# In[13]:


from sklearn import metrics
nodes = cluster_info
average_score = metrics.silhouette_score(a, nodes, metric='precomputed')
print(average_score)

score_samples = metrics.silhouette_samples(a, nodes, metric='precomputed')
df['silhouette_coeff'] = score_samples

df.groupby('cluster')['silhouette_coeff'].mean()

# df.to_csv('5min_seq_trimmed_clst.csv')


# In[ ]:





# In[ ]:





# In[14]:


clst1 = []
clst2 = []
clst3 = []
clst4 = []
clst5 = []
clst6 = []
clst7 = []
clst8 = []
clst9 = []
clst10 = []
clst11 = []
clst12 = []
clst13 = []

for i in range(len(cluster_info)):
    if cluster_info[i] == 1:
        clst1.append(i)
    elif cluster_info[i] == 2:
        clst2.append(i)
    elif cluster_info[i] == 3:
        clst3.append(i)
    elif cluster_info[i] == 4:
        clst4.append(i)
    elif cluster_info[i] == 5:
        clst5.append(i)
    elif cluster_info[i] == 6:
        clst6.append(i)
    elif cluster_info[i] == 7:
        clst7.append(i)
    elif cluster_info[i] == 8:
        clst8.append(i)
    elif cluster_info[i] == 9:
        clst9.append(i)
    elif cluster_info[i] == 10:
        clst10.append(i)
    elif cluster_info[i] == 11:
        clst11.append(i)
    elif cluster_info[i] == 12:
        clst12.append(i)
    elif cluster_info[i] == 13:
        clst13.append(i)


print(len(clst1))
print(len(clst2))
print(len(clst3))
print(len(clst4))
print(len(clst5))
print(len(clst6))
print(len(clst7))
print(len(clst8))
print(len(clst9))
print(len(clst10))
print(len(clst11))
print(len(clst12))
print(len(clst13))


# In[21]:


def visualize_silhouette(clustering, X_features, cluster_lists=[1]): 
    
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(8*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        cluster_labels = cluster_info
        
        #if clustering[0] == 'dbscan':
        #    n_cluster = len(set(cluster_labels))-1
        
        sil_avg = metrics.silhouette_score(data, cluster_labels, metric='precomputed')
        sil_values = metrics.silhouette_samples(data, cluster_labels, metric='precomputed')
        
        y_lower = 10
        axs[ind].set_title("The silhouette plot for the various clusters. - Hierarchical",fontsize=15)
        axs[ind].set_xlabel("The silhouette coefficient values",fontsize=20)
        axs[ind].set_ylabel("Cluster label",fontsize=20)
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([-1, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values,                                 facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i),fontsize=15)
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
        
        plt.savefig('hie_cos_sil.png')


# In[16]:


from sklearn.cluster import AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=13, linkage='complete', affinity="cosine")
labels = agg_clustering.fit_predict(data)


# In[17]:


from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import math

score = silhouette_score(data,labels)
print(score)


# In[22]:


visualize_silhouette(['hierarchical', 'ward'], data, [2,13])


# In[ ]:




