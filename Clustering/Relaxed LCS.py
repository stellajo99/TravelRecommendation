#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import requests
import pandas as pd
import numpy as np
import copy

df=pd.read_csv('type_clust1.csv')
df['trimmed'] = df.trimmed.apply(lambda x: x[1:-1].split(','))
arr=df.trimmed
df2=pd.read_csv('type_clust2.csv')
df2['trimmed'] = df2.trimmed.apply(lambda x: x[1:-1].split(','))
arr2=df2.trimmed
df3=pd.read_csv('type_clust3.csv')
df3['trimmed'] = df3.trimmed.apply(lambda x: x[1:-1].split(','))
arr3=df3.trimmed
df4=pd.read_csv('type_clust4.csv')
df4['trimmed'] = df4.trimmed.apply(lambda x: x[1:-1].split(','))
arr4=df4.trimmed
df5=pd.read_csv('type_clust5.csv')
df5['trimmed'] = df5.trimmed.apply(lambda x: x[1:-1].split(','))
arr5=df.trimmed
df6=pd.read_csv('type_clust6.csv')
df6['trimmed'] = df6.trimmed.apply(lambda x: x[1:-1].split(','))
arr6=df6.trimmed


# Function to find lcs_algo
def lcs_algo(S1, S2, m, n):
    L = [[0 for x in range(n+1)] for x in range(m+1)]

    seq1=copy.deepcopy(S1)
    seq2=copy.deepcopy(S2)
    
    for i in range(m):
        if i==0:
            seq1[i]=seq1[i][2:8]
        else:
            seq1[i]=seq1[i][4:10]
        #print(seq1[i])
    
    for i in range(n):
        if i==0:
            seq2[i]=seq2[i][2:8]
        else:
            seq2[i]=seq2[i][4:10]
        #print(seq2[i])
    

    # Building the mtrix in bottom-up way
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif seq1[i-1] == seq2[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    index = L[m][n]

    lcs_algo = [""] * (index+1)
    lcs_algo[index] = ""

    i = m
    j = n
    while i > 0 and j > 0:

        if seq1[i-1] == seq2[j-1]:
            lcs_algo[index-1] = seq1[i-1]
            i -= 1
            j -= 1
            index -= 1

        elif L[i-1][j] > L[i][j-1]:
            i -= 1
        else:
            j -= 1
            
    #print(lcs_algo)
    if m>n:
        if seq1[0] not in lcs_algo:
            lcs_algo.insert(0,seq1[0])
            if seq1[m-1] not in lcs_algo:
                lcs_algo.insert(len(lcs_algo)-1,seq1[m-1])
    elif m<n:
        if seq2[0] not in lcs_algo:
            lcs_algo.insert(0,seq2[0])
            if seq2[n-1] not in lcs_algo:
                lcs_algo.insert(len(lcs_algo)-1, seq2[n-1])
    # Printing the sub sequences
    #print("S1 : ",end='')
    #print(seq1)
    #print("S2 : ",end='')
    #print(seq2)
    #print("LCS: " + "".join(lcs_algo))
    return lcs_algo


    
lcs_algo(arr[1],arr[2],len(arr[1]),len(arr[2]))
lcs_algo(arr[1],arr[3],len(arr[1]),len(arr[3]))

#for i in range(4):
#    for j in range(4):
#        if i==j:
#            print('')
#        else:
#            lcs_algo(arr[i],arr[j],len(arr[i]),len(arr[j]))
        
#        print('')


# In[26]:


import datetime
import requests
import pandas as pd
import numpy as np
import copy

df=pd.read_csv('type_clust1.csv')
df['trimmed'] = df.trimmed.apply(lambda x: x[1:-1].split(','))
arr=df.trimmed
df2=pd.read_csv('type_clust2.csv')
df2['trimmed'] = df2.trimmed.apply(lambda x: x[1:-1].split(','))
arr2=df2.trimmed
df3=pd.read_csv('type_clust3.csv')
df3['trimmed'] = df3.trimmed.apply(lambda x: x[1:-1].split(','))
arr3=df3.trimmed
df4=pd.read_csv('type_clust4.csv')
df4['trimmed'] = df4.trimmed.apply(lambda x: x[1:-1].split(','))
arr4=df4.trimmed
df5=pd.read_csv('type_clust5.csv')
df5['trimmed'] = df5.trimmed.apply(lambda x: x[1:-1].split(','))
arr5=df.trimmed
df6=pd.read_csv('type_clust6.csv')
df6['trimmed'] = df6.trimmed.apply(lambda x: x[1:-1].split(','))
arr6=df6.trimmed


# Function to find lcs_algo
def lcs_algo_arr(S1, S2, m, n):
    L = [[0 for x in range(n+1)] for x in range(m+1)]

    seq1=copy.deepcopy(S1)
    seq2=copy.deepcopy(S2)
    
    for i in range(n):
        if i==0:
            seq2[i]=seq2[i][2:8]
        else:
            seq2[i]=seq2[i][4:10]
        #print(seq2[i])
    

    # Building the mtrix in bottom-up way
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif seq1[i-1] == seq2[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    index = L[m][n]

    lcs_algo = [""] * (index+1)
    lcs_algo[index] = ""

    i = m
    j = n
    while i > 0 and j > 0:

        if seq1[i-1] == seq2[j-1]:
            lcs_algo[index-1] = seq1[i-1]
            i -= 1
            j -= 1
            index -= 1

        elif L[i-1][j] > L[i][j-1]:
            i -= 1
        else:
            j -= 1
            
    #print(lcs_algo)
    if m>n:
        if seq1[0] not in lcs_algo:
            lcs_algo.insert(0,seq1[0])
            if seq1[m-1] not in lcs_algo:
                lcs_algo.insert(len(lcs_algo)-1,seq1[m-1])
    elif m<n:
        if seq2[0] not in lcs_algo:
            lcs_algo.insert(0,seq2[0])
            if seq2[n-1] not in lcs_algo:
                lcs_algo.insert(len(lcs_algo)-1, seq2[n-1])
    # Printing the sub sequences
    #print("S1 : ",end='')
    #print(seq1)
    #print("S2 : ",end='')
    #print(seq2)
    #print("LCS: " + "".join(lcs_algo))
    return lcs_algo


# In[ ]:




