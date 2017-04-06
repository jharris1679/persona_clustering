
# coding: utf-8

# # TODO
# 
# - write function for undo zscore to see cluster centers in terms of the value not the zscore (this would be pretty great and not hard with a bit of focus)
#  - X = Z * std - mean
# - write function for creating a pdf for bonus points

# In[2]:

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from pandas.tools.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.switch_backend('MacOSX') 


# In[104]:

csv = pd.read_csv("amb_selec3.csv") #import the file of users and their features, pulled from BigQuery
fs = '3' #feature set version

csv = csv.loc[csv['recd_praises']>1,].dropna().reset_index(drop=True) # threshold on feature, drop rows with na values
strip = csv[(np.abs(stats.zscore(csv)) < 3).all(axis=1)].reset_index(drop=True) # remove crazy outliers
drop = pd.concat([pd.DataFrame(strip.iloc[:,1:3]), pd.DataFrame(strip.iloc[:,4:7])], axis = 1, join_axes = [strip.index])# drop the user_ids and any other features you wish to discard
data = pd.DataFrame(scale(drop)) # use this for < 4D
#data3D = pd.DataFrame(PCA(n_components=3).fit_transform(data)) # Reduce dimensions from 4 to 3 for visualization
data2D = pd.DataFrame(PCA(n_components=2).fit_transform(data)) # Reduce dimensions from 4 to 2 for visualization

# run kmeans with 10 sets of clusters seeds
# extract classification labels and cluster centers 
# do some prep for vizualization
def cluster_it_up(N):
    kmeans = KMeans(init = 'k-means++', n_clusters = N, n_init = 100).fit(data) # it's as easy as a function call
    labels = pd.DataFrame((kmeans.labels_)) # extract labels
    centers = pd.DataFrame(kmeans.cluster_centers_) # extract centers
    centers.columns = drop.columns # give centers table readable column names
    labels.columns = ['label'] 
    clust_out = pd.concat([strip, labels], axis=1, join_axes = [strip.index]) # make DataFrame that is labeled users and their features
    viz3D = pd.concat([data, labels], axis = 1, join_axes = [data.index]) 
    viz2D = pd.concat([data2D, labels], axis = 1, join_axes = [data2D.index])
    return clust_out, centers, N, strip, data, labels, viz3D, viz2D


# In[107]:

clust_out.to_csv('~/Google Drive/ambassador_selection/' + str(N) + '_clus_fs' + fs + '.csv')


# In[105]:

clust_out, centers, N, strip, data, labels, viz3D, viz2D  = cluster_it_up(5)


# In[106]:

get_ipython().magic('matplotlib inline')
colors = plt.cm.rainbow(np.linspace(0, 1, len(centers)))

#2D viz
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
for i,c in enumerate(colors):
    single_clus = viz2D.loc[viz2D['label'] == i]
    plt.scatter(single_clus[0], single_clus[1], s=3, c = c, label=str(i))
plt.legend()
plt.show()

#3D viz
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
ax = fig.add_subplot(111, projection='3d')
for i,c in enumerate(colors):
    single_clus = viz3D.loc[viz3D['label'] == i]
    ax.scatter(single_clus[1], single_clus[2], single_clus[0], s=20, c = c, label=str(i))
plt.legend()
plt.show()

#scatter matrix
#scatter_matrix(data, alpha=0.1, figsize=(6, 6), diagonal='kde')
#plt.show()

centers['sum'] = centers.sum(axis = 1)
centers


# In[89]:

data.shape


# In[96]:

good_clus = pd.DataFrame(clust_out.loc[clust_out['label'] == 1]).reset_index(drop = True) 
good_clus.sort_values(by = ['praise_length'], ascending = [False], inplace = True)
good_clus


# In[ ]:



