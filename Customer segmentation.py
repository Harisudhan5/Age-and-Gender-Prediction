#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import math
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# In[2]:


customers=pd.read_csv('C:/Users/HARISUDHAN/Downloads/Customer-Segmentation-using-K-means-master/Customer-Segmentation-using-K-means-master/Mall_Customers.csv') 
print("Shape of data=>",customers.shape)


# In[3]:


customers.head(16)


# In[4]:


print(f"Missing values in each variable: \n{customers.isnull().sum()}")


# In[5]:


print(f"Duplicated rows: {customers.duplicated().sum()}")


# In[6]:


customers.dtypes


# In[7]:


def statistics(variable):
    if variable.dtype == "int64" or variable.dtype == "float64":
        return pd.DataFrame([[variable.name, np.mean(variable), np.std(variable), np.median(variable), np.var(variable)]], 
                            columns = ["Variable", "Mean", "Standard Deviation", "Median", "Variance"]).set_index("Variable")
    else:
        return pd.DataFrame(variable.value_counts())


# In[8]:


def graph_histo(x):
    if x.dtype == "int64" or x.dtype == "float64":
        # Select size of bins by getting maximum and minimum and divide the substraction by 10
        size_bins = 10
        # Get the title by getting the name of the column
        title = x.name
        #Assign random colors to each graph
        color_kde = list(map(float, np.random.rand(3,)))
        color_bar = list(map(float, np.random.rand(3,)))

        # Plot the displot
        sns.distplot(x, bins=size_bins, kde_kws={"lw": 1.5, "alpha":0.8, "color":color_kde},
                       hist_kws={"linewidth": 1.5, "edgecolor": "grey",
                                "alpha": 0.4, "color":color_bar})
        # Customize ticks and labels
        plt.xticks(size=14)
        plt.yticks(size=14);
        plt.ylabel("Frequency", size=16, labelpad=15);
        # Customize title
        plt.title(title, size=18)
        # Customize grid and axes visibility
        plt.grid(False);
        plt.gca().spines["top"].set_visible(False);
        plt.gca().spines["right"].set_visible(False);
        plt.gca().spines["bottom"].set_visible(False);
        plt.gca().spines["left"].set_visible(False); 
    else:
        x = pd.DataFrame(x)      
        sns.catplot(x=x.columns[0], kind="count", palette="spring", data=x)
        # Customize title
        title = x.columns[0]
        plt.title(title, size=18)
        # Customize ticks and labels
        plt.xticks(size=14)
        plt.yticks(size=14);
        plt.xlabel("")
        plt.ylabel("Counts", size=16, labelpad=15)        
        # Customize grid and axes visibility
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)


# In[9]:


spending = customers["Spending Score (1-100)"]


# In[10]:


statistics(spending)


# In[11]:


graph_histo(spending)


# In[12]:


age = customers["Age"]
statistics(age)


# In[13]:


graph_histo(age)


# In[14]:


income = customers["Annual Income (k$)"]


# In[15]:


statistics(income)


# In[16]:


graph_histo(income)


# In[17]:


gender = customers["Gender"]


# In[18]:


statistics(gender)


# In[19]:


graph_histo(gender)


# In[20]:


customers["Male"] = customers.Gender.apply(lambda x: 0 if x == "Male" else 1)


# In[21]:


customers["Female"] = customers.Gender.apply(lambda x: 0 if x == "Female" else 1)


# In[22]:


X = customers.iloc[:, 2:]


# In[23]:


X.head(12)


# In[24]:


pca = PCA(n_components=2).fit(X)


# In[25]:


print(pca.components_)


# In[26]:


print(pca.explained_variance_)


# In[27]:


pca_2d = pca.transform(X)


# In[28]:


wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss, c="#c51b7d")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.title('Elbow Method', size=14)
plt.xlabel('Number of clusters', size=12)
plt.ylabel('wcss', size=14)


# In[29]:


kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=10, n_init=10, random_state=0)
y_means = kmeans.fit_predict(X)


# In[30]:


fig, ax = plt.subplots(figsize = (8, 6))

plt.scatter(pca_2d[:, 0], pca_2d[:, 1],
            c=y_means, 
            edgecolor="none", 
            cmap=plt.cm.get_cmap("Spectral_r", 5),
            alpha=0.5)
        
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)

plt.xticks(size=12)
plt.yticks(size=12)

plt.xlabel("Component 1", size = 14, labelpad=10)
plt.ylabel("Component 2", size = 14, labelpad=10)

plt.title('Domains Grouped in 5 clusters', size=16)


plt.colorbar(ticks=[0, 1, 2, 3, 4]);

plt.show()


# In[31]:


centroids = pd.DataFrame(kmeans.cluster_centers_, columns = ["Age", "Annual Income", "Spending", "Male", "Female"])


# In[32]:


centroids.index_name = "ClusterID"
centroids["ClusterID"] = centroids.index
centroids = centroids.reset_index(drop=True)


# In[33]:


centroids


# In[35]:


X_new = np.array([[22,10,9,0,2]])  # enter the attributes to get segments for new customer 
new_customer = kmeans.predict(X_new)
print(f"The new customer belongs to segment {new_customer[0]}")


# In[ ]:




