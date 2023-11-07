#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv(r"C:\Users\lalit\OneDrive\Desktop\practicals\machine learning\practical3\diabetes.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.head(20)


# In[7]:


df.isnull()


# In[10]:


df.shape


# In[11]:


df.dtypes


# In[12]:


df.isnull().sum()


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


plt.figure(figsize=(10, 6))
sns.boxplot(x="Outcome", y="Glucose", data=df)
plt.xlabel("Outcome")
plt.ylabel("Glucose")
plt.show()


# In[31]:


sns.countplot(x="Outcome", data=df)
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()


# In[14]:


# Assuming your data is in a DataFrame called 'data'
X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Pedigree", "Age"]]
y = df["Outcome"]


# In[15]:


from sklearn.model_selection import train_test_split
# Split the data into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier

# Choose the value of K (number of neighbors)
k = 5
knn = KNeighborsClassifier(n_neighbors=k)


# In[17]:


# Fit the KNN model to the training data
knn.fit(X_train, y_train)


# In[18]:


# Make predictions on the test data
y_pred = knn.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[22]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[23]:


# Calculate error rate
error_rate = 1 - accuracy
print(f"Error Rate: {error_rate:.2f}")


# In[24]:


# Calculate precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")


# In[25]:


# Calculate recall
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.2f}")


# In[ ]:




