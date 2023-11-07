#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[7]:


df = pd.read_csv(r"C:\Users\lalit\OneDrive\Desktop\practicals\machine learning\practical4\sales_data_sample.csv")


# In[8]:


df.head()


# In[9]:


df.head(50)


# In[10]:


df.info()


# In[11]:


df.shape


# In[12]:


df.isnull().sum()


# In[13]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[21]:



# Select relevant numerical columns for clustering
numerical_columns = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES']]


# In[23]:


# Determine the optimal number of clusters using the elbow method
wcss = []  # Within-Cluster-Sum-of-Squares


# In[24]:


# Iterate over a range of values for k (number of clusters)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(numerical_columns)  # Fit K-Means to the data
    wcss.append(kmeans.inertia_)  # Append the inertia value to the list


# In[25]:


# Plot the elbow graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster-Sum-of-Squares)')
plt.show()




# In[26]:


# Based on the elbow method, let's say the optimal number of clusters is 3
optimal_k = 3


# In[29]:


# Perform K-Means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(numerical_columns)


# In[37]:


from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

numerical_columns = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES']]


# In[38]:


# Create a linkage matrix using the Ward method
Z = linkage(numerical_columns, method='ward')


# In[39]:


# Plot the hierarchical clustering dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, p=optimal_k, truncate_mode='lastp')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()


# In[40]:


from scipy.cluster.hierarchy import fcluster


# In[41]:


# Based on the dendrogram, let's say the optimal number of clusters is 3
optimal_k = 3


# In[42]:


# Cut the dendrogram to obtain cluster assignments
cluster_assignments = fcluster(Z, t=optimal_k, criterion='maxclust')


# In[44]:


# Add the cluster assignments to your original dataset
df['cluster'] = cluster_assignments 


# In[47]:


df.head()


# In[ ]:




