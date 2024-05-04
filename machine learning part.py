#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing various libraries of python
import numpy as np #For array mathematics
import pandas as pd #for data manipulation
import matplotlib.pyplot as plt #for data visualistion
import seaborn as sns #for data visualistion
from datetime  import date #for time
from sklearn.cluster import KMeans #for machine learning algorithms


# In[2]:


#importing the excel dataset
filepath=r"C:\Users\Riya\Desktop\Online Retail (1).xlsx"
df = pd.read_excel(filepath)


# In[3]:


#how does data look (displaying 10 columns of dataset)
df.sample(10)


# In[4]:


#number of rows and coloums in dataset
df.shape


# In[5]:


#finding information of data
df.info()


# In[6]:


#describing numerical columns of dataset
df.describe()


# In[7]:


#finding correlations between numerical columns
df.corr()


# In[8]:


#finding if null values are present or not
df.isnull().sum()


# In[9]:


#plotting heatmap (it is used to represent null values )
missing_value=df.isnull()
plt.figure(figsize=(8,8))
sns.heatmap(missing_value)
plt.show()


# In[10]:


#remving null values permanently
df.dropna(subset=["CustomerID"],inplace=True)


# In[11]:


#checking if null values removed
df.isnull().sum()


# In[12]:


#finding if duplicate values are present or not
df.duplicated().sum()


# In[13]:


#removing the duplicates permanently
df.drop_duplicates(inplace=True)


# In[14]:


df.info()


# In[15]:


#plotting heatmap after removing null values
processed_value=df.isnull()
plt.figure(figsize=(8,8))
sns.heatmap(processed_value)
plt.show()


# In[16]:


#finding negative values in quantity column (i.e the products which are returned)
df[df["Quantity"]<0]


# In[17]:


#updating the dataset with only positive values
df=df[df["Quantity"]>0]
df


# In[18]:


df.info()


# In[19]:


df["Country"].unique()


# In[20]:


#plotting sales for top 10 countries
#index is reset as many entries are removed
country_data=df.groupby("Country").count().reset_index()
country_data.sort_values("InvoiceNo",ignore_index=True,ascending=False,inplace=True)
fig,axes=plt.subplots(figsize=(16,8))
sns.set_style("darkgrid")
sns.barplot(data=country_data[0:10],x="Country",y="InvoiceNo",ax=axes,linewidth=1)
axes.set_yticks(range(0,400000,20000))
axes.set_xlabel("Country", size=20)
axes.set_ylabel("Number of Transactions", size=20)
axes.set_title("Transactions per Country",size=30)


# In[21]:


df.columns


# In[22]:


df["Month"]=df["InvoiceDate"].apply(lambda x:x.month)
df.head()


# In[23]:


df["Month"].unique()


# In[24]:


Month_data=df.groupby("Month").count().reset_index()
fig,axes=plt.subplots(figsize=(16,8))
sns.set_style("darkgrid")
sns.barplot(data=Month_data,x="Month",y="InvoiceNo",ax=axes,linewidth=1)
axes.set_yticks(range(0,80000,10000))
axes.set_xlabel("Month", size=20)
axes.set_ylabel("Number of Transactions", size=20)
axes.set_title("Transactions per Month",size=30)


# In[25]:


#getting the monetary value of each customer from quantity and unit price
df["MonetaryValue"]=df.apply(lambda x:x["Quantity"]*x["UnitPrice"],axis=1)
df.head()


# In[26]:


Customer_data=df.groupby("CustomerID").sum().reset_index()
Customer_data


# In[27]:


sns.histplot(Customer_data["MonetaryValue"])
plt.show()


# In[28]:


Customer_data.drop(columns=["Quantity","UnitPrice","Month"],inplace=True)
Customer_data


# In[29]:


# frequency:how many times a customer purchased in particular duration(respresented by f)
# recency:when the  customer purchased recently(respresented by r)
# monetary value:total money spent(respresented by m)
Customer_data["Frequency"]=df.groupby("CustomerID")["MonetaryValue"].count().values
Customer_data


# In[30]:


sns.histplot(Customer_data["Frequency"])


# In[31]:


#last date is 09/12/2011
last_date=date(2011,12,10)
df["Recency"]=df["InvoiceDate"].apply(lambda x:(last_date-pd.to_datetime(x).date()).days)
df


# In[32]:


Customer_data["Recency"]=df.groupby("CustomerID")["Recency"].min().values
Customer_data


# In[33]:


#plotting the recency
sns.histplot(Customer_data["Recency"])


# # Data preprocessing

# In[34]:


#applying log transformation
Customer_data["MonetaryValue"]=Customer_data["MonetaryValue"].apply(lambda x:np.log(x+1)) #adding 1 for free item
Customer_data["Frequency"]=Customer_data["Frequency"].apply(lambda x:np.log(x))
Customer_data["Recency"]=Customer_data["Recency"].apply(lambda x:np.log(x))
Customer_data


# In[35]:


#plotting the data log transformation
fig,axis = plt.subplots(nrows=1,ncols=3,figsize=(20,8))
sns.histplot(Customer_data["Recency"],ax=axis[0])
sns.histplot(Customer_data["Frequency"],ax=axis[1])
sns.histplot(Customer_data["MonetaryValue"],ax=axis[2])
fig.suptitle("Data distribution after log transformation",size=25)


# In[36]:


#! pip install plotly


# In[37]:


#import plotly.express as px
#fig=px.scatter_3d(Customer_data, x="Recency", y="Frequency", z="MonetaryValue")
#fig.show()


# In[38]:


data=Customer_data.drop(columns=["CustomerID"])
data


# In[39]:


#calculating elbow method score
sse={}
for k in range(1,21):
    kmeans=KMeans(n_clusters=k,random_state=1)
    kmeans.fit(data)
    sse[k]=kmeans.inertia_


# In[40]:


sse


# In[41]:


#plotting elbow
plt.figure(figsize=(12,8))
plt.title("The Elbow Method",size=25)
plt.xlabel("number of clusters",size=20)
plt.ylabel("sum of squared error",size=20)
sns.pointplot(x=list(sse.keys()),y=list(sse.values()))
plt.show()


# In[42]:


cluster=KMeans(n_clusters=4,random_state=1)
cluster_label=cluster.fit_predict(data)
cluster_label


# In[43]:


#adding cluster label to each data point
Customer_data["cluster"]=cluster_label
Customer_data


# In[44]:


#calculating the mean of each feature for each cluster
cluster_data=Customer_data.groupby("cluster").mean()
cluster_data


# In[45]:


cluster_data.drop(columns=["CustomerID"], inplace=True)


# In[46]:


#we have log transformed the features to get more intution take expontiation of each feature
cluster_data=cluster_data.applymap(np.exp)
cluster_data=cluster_data.applymap(int)
cluster_data


# In[48]:


import pickle
pickle_out=open("cluster.pkl","wb")
pickle.dump(cluster,pickle_out)
pickle_out.close()


# In[ ]:





# In[ ]:




