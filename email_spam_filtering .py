#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[3]:


df = pd.read_csv(r"C:\Users\lalit\OneDrive\Desktop\practicals\machine learning\practical2\emails.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


#Removing the null values
df.isnull().sum()


# In[16]:


from sklearn.model_selection import train_test_split

# Define X (Features)
X = df.drop(columns=['Email No.', 'Prediction'])

# Define y (Target)
y = df['Prediction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Create a K-NN model
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors


# In[22]:


# Train the K-NN model on the training data
knn_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_knn = knn_model.predict(X_test)


# In[26]:


# Evaluate the K-NN model's performance
print("K-Nearest Neighbors (K-NN) Performance:")


# In[27]:



accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"Accuracy: {accuracy_knn:.2f}")


# In[28]:


precision_knn = precision_score(y_test, y_pred_knn)
print(f"Precision: {precision_knn:.2f}")


# In[29]:


recall_knn = recall_score(y_test, y_pred_knn)
print(f"Recall: {recall_knn:.2f}")


# In[30]:


f1_score_knn = f1_score(y_test, y_pred_knn)
print(f"F1 Score: {f1_score_knn:.2f}")


# In[32]:


roc_auc_knn = roc_auc_score(y_test, y_pred_knn)
print(f"ROC-AUC: {roc_auc_knn:.2f}")


# In[33]:


from sklearn.svm import SVC

# Create an SVM model
svm_model = SVC(kernel='linear', C=1)  # You can adjust the kernel and C parameter

# Train the SVM model on the training data
svm_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_svm = svm_model.predict(X_test)


# In[35]:



# Evaluate the SVM model's performance
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_score_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)

# Print the SVM model's evaluation metrics
print("\nSupport Vector Machine (SVM) Performance:")
print(f"Accuracy: {accuracy_svm:.2f}")
print(f"Precision: {precision_svm:.2f}")
print(f"Recall: {recall_svm:.2f}")
print(f"F1 Score: {f1_score_svm:.2f}")
print(f"ROC-AUC: {roc_auc_svm:.2f}")


# In[ ]:




