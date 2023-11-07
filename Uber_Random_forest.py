#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np


# In[3]:


df = pd.read_csv(r"C:\Users\lalit\OneDrive\Desktop\practicals\machine learning\practical1\uber.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df= df.drop(['Unnamed: 0', 'key'], axis= 1) #To drop unnamed column as it isn't required 


# In[8]:


df.head(50)


# In[9]:


df.shape


# In[10]:


df.dtypes


# In[11]:


#Removing the null values
df.isnull().sum()


# In[12]:


df['dropoff_latitude'].fillna(value=df['dropoff_latitude'].mean(),inplace= True)
df['dropoff_longitude'].fillna(value=df['dropoff_longitude'].median(),inplace= True)


# In[13]:


df.isnull().sum()


# In[14]:


import matplotlib.pyplot as plt


# In[16]:


plt.boxplot(df['fare_amount'])
plt.title("Box Plot of Fare Amount")
plt.show()


# In[19]:


# Use IQR method to detect and remove outliers
Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)
IQR = Q3 - Q1
data = df[~((df['fare_amount'] < (Q1 - 1.5 * IQR)) | (df['fare_amount'] > (Q3 + 1.5 * IQR)))]  # Remove outliers


# In[32]:


X= df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count' ]]
y= df['fare_amount']


# In[33]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[34]:


# Train a Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[35]:


# Train a Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[38]:


# 5. Evaluate the models and compare their respective scores
# Make predictions
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)


# In[39]:


from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt


# In[40]:


# Calculate R-squared (R2)
lr_r2 = r2_score(y_test, lr_predictions)
rf_r2 = r2_score(y_test, rf_predictions)


# In[41]:


# Calculate Root Mean Squared Error (RMSE)
lr_rmse = sqrt(mean_squared_error(y_test, lr_predictions))
rf_rmse = sqrt(mean_squared_error(y_test, rf_predictions))


# In[42]:


print("Linear Regression R2:", lr_r2)
print("Random Forest Regression R2:", rf_r2)
print("Linear Regression RMSE:", lr_rmse)
print("Random Forest Regression RMSE:", rf_rmse)


# In[ ]:




