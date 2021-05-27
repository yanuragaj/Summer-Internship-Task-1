#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas


# In[2]:


dataset = pandas.read_csv('Salary_Data.csv')


# In[3]:


dataset


# In[4]:


dataset.columns


# In[5]:


y=dataset['Salary']


# In[6]:


x=dataset['YearsExperience']


# In[7]:


y


# In[8]:


x


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


model=LinearRegression()


# In[11]:


x=x.values


# In[12]:


x


# In[13]:


x=x.reshape(-1,1)


# In[14]:


model.fit(x,y)


# In[15]:


model.predict([[10.1]])


# In[16]:


model.coef_


# In[17]:


model.intercept_


# In[21]:


model.predict([[10.]])


# In[19]:


import joblib


# In[20]:


joblib.dump(model,'salary.pk1') # will create a salary.pk1 file with above trained module 

