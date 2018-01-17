
# coding: utf-8

# In[19]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist, pdist



# In[20]:

studentNormData = np.random.uniform(-1,1,size=(100,4))
#studentsDf = pd.DataFrame(data=studentsData)
#studentNormData = scale(studentsDf)


# In[21]:

estimators = [('studentData_3_clusters', KMeans(n_clusters=3, n_init = 3)), ('studentData_4_clusters', KMeans(n_clusters=4, n_init = 3)), ('studentData_5_clusters', KMeans(n_clusters=5, n_init = 3)), ('studentData_6_clusters', KMeans(n_clusters=6, n_init = 3)), ('studentData_7_clusters', KMeans(n_clusters=7, n_init = 3)), ('studentData_8_clusters', KMeans(n_clusters=8, n_init = 3))]


# In[22]:

fignum = 1


# In[23]:

titles = ['3 clusters', '4 clusters', '5 clusters', '6 clusters', '7 clusters', '8 clusters']
get_ipython().magic('matplotlib inline')
sse = []
centroids2 = []
D_k = []


# In[24]:

fignum = 1
range_n_clusters = [3, 4, 5, 6, 7, 8]
for name, est in estimators:
    n_clusters = range_n_clusters[fignum - 1]
    #fig = plt.figure(fignum, figsize=(4, 3))
    
    est.fit(studentNormData)
    labels = est.labels_
    
    
    # Create a subplot with 1 row and 1 columns
    sfig, ax1 = plt.subplots(1, 1)
    sfig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -0.4, 1 
    ax1.set_xlim([-0.4, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(studentNormData) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(studentNormData, est.labels_)
    print("For n_clusters =", titles[fignum - 1], "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(studentNormData, est.labels_)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[est.labels_ == i]
        print("Population ", len(ith_cluster_silhouette_values))
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    #ax2 = Axes3D(sfig, rect=[1, 0, 0.95, 1], elev=48, azim=130)
    
   # ax2.scatter(studentNormData[:, 0], studentNormData[:, 1], studentNormData[:, 2], c=labels.astype(np.float), edgecolor='k')

    centroids = est.cluster_centers_
    #centroids2.append = est.cluster_centers_
    #D_k.append = cdist(studentNormData, centroids, 'euclidean')
    #ax2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=169, marker='x', linewidths=7, color='b')

    #ax2.w_xaxis.set_ticklabels([])
    #ax2.w_yaxis.set_ticklabels([])
    #ax2.w_zaxis.set_ticklabels([])
    #ax2.set_xlabel('Physics')
    #ax2.set_ylabel('Maths')
    #ax2.set_zlabel('English')
    print("Sum of square errors = ", est.inertia_)
    print("Cluster Centers ", est.cluster_centers_)
    #print("Centroids ", centroids)
    print(" \n")
    sse.append(est.inertia_)
    #ax2.set_title(titles[fignum - 1])
    #ax2.dist = 12

    fignum = fignum + 1


# In[25]:

plt.plot(range(3,9), sse, 'b*-')
plt.grid(True)
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared errors')


# In[24]:

fignum = 1
range_n_clusters = [3, 4, 5, 6, 7, 8]
for name, est in estimators:
    n_clusters = range_n_clusters[fignum - 1]
    #fig = plt.figure(fignum, figsize=(4, 3))
    
    est.fit(studentNormData[:, [1, 2, 3]])
    labels = est.labels_
    
    
    # Create a subplot with 1 row and 1 columns
    sfig, ax1 = plt.subplots(1, 1)
    sfig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -0.4, 1 
    ax1.set_xlim([-0.4, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(studentNormData[:, [1, 2, 3]]) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(studentNormData[:, [1, 2, 3]], est.labels_)
    print("For n_clusters =", titles[fignum - 1], "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(studentNormData[:, [1, 2, 3]], est.labels_)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[est.labels_ == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    ax2 = Axes3D(sfig, rect=[1, 0, 0.95, 1], elev=48, azim=130)
    
    ax2.scatter(studentNormData[:, 1], studentNormData[:, 2], studentNormData[:, 3], c=labels.astype(np.float), edgecolor='k')

    centroids = est.cluster_centers_
    ax2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=169, marker='x', linewidths=7, color='b')

    ax2.w_xaxis.set_ticklabels([])
    ax2.w_yaxis.set_ticklabels([])
    ax2.w_zaxis.set_ticklabels([])
    ax2.set_xlabel('Maths')
    ax2.set_ylabel('English')
    ax2.set_zlabel('Music')
    print(est.inertia_)
    print(est.cluster_centers_)
    print(centroids[:, 2])
    sse.append(est.inertia_)
    ax2.set_title(titles[fignum - 1])
    ax2.dist = 12

    fignum = fignum + 1


# In[25]:

studentNormData


# In[27]:

est.labels_


# In[ ]:



