
# coding: utf-8

# In[103]:


# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()  # for plot styling


from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score


# In[104]:

get_ipython().magic('matplotlib inline')
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation


# In[105]:

address = 'D:/Meng Classes/Intelligent data analysis/Homework 3/HW3-StudentData3.xlsx'
studentsData = pd.read_excel(address)
studentsDf = pd.DataFrame(data=studentsData)
studentNormData = scale(studentsDf)
studentPDDist = pdist(studentNormData, 'euclidean')


# In[106]:

print(studentNormData.shape)


# In[107]:

Z = linkage(studentNormData, 'single', 'euclidean')


# In[108]:

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[109]:

A = linkage(studentNormData, 'complete', 'euclidean')
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    A,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[69]:

completeLink = fcluster(linkage(studentNormData, 'complete', 'euclidean'), 4, 'maxclust')
completeLink


# In[70]:

singleLink = fcluster(linkage(studentNormData, 'single', 'euclidean'), 4, 'maxclust')
singleLink


# In[71]:

#cluster_output = pandas.DataFrame({'team':studentNormData. , 'cluster':assignments})
adjusted_rand_score(completeLink, singleLink)


# In[72]:

#studentDfNorm = pd.concat(studentNormData, assignments)


# In[73]:

studentDfNorm = pd.DataFrame(data=studentNormData)


# In[ ]:




# In[ ]:




# In[75]:

singleLinkClusterAssignments = pd.DataFrame(data=singleLink)
completeLinkClusterAssignments = pd.DataFrame(data=completeLink)


# In[125]:

singleLinkStudentClustNorm = pd.concat([studentDfNorm, singleLinkClusterAssignments], axis=1)
completeLinkStudentClustNorm = pd.concat([studentDfNorm, completeLinkClusterAssignments], axis=1)
singleLinkStudentClustNorm


# In[77]:

studentClustNorm.head()


# In[78]:

studentId= range(1,70)


# In[79]:

singleLinkStudentClustNorm.columns = ['physics', 'maths', 'english', 'music', 'clusters']
completeLinkStudentClustNorm.columns = ['physics', 'maths', 'english', 'music', 'clusters']


# In[80]:

studentClustNorm.head()


# In[ ]:




# In[ ]:




# In[81]:

studentId


# In[82]:

studentId = pd.DataFrame(np.arange(1,70).reshape(69,1))


# In[83]:

studentId.head()


# In[84]:

singleLinkStudentClustNorm = pd.concat([studentId, singleLinkStudentClustNorm], axis=1)
completeLinkStudentClustNorm = pd.concat([studentId, completeLinkStudentClustNorm], axis=1)


# In[85]:

singleLinkStudentClustNorm.head()


# In[86]:

singleLinkStudentClustNorm.columns = ['studentId', 'physics', 'maths', 'english', 'music', 'clusters']
completeLinkStudentClustNorm.columns = ['studentId', 'physics', 'maths', 'english', 'music', 'clusters']


# In[87]:

completeLinkStudentClustNorm.head()


# In[88]:

singleLinkStudentGroup = singleLinkStudentClustNorm.groupby(singleLinkStudentClustNorm['clusters'])
completeLinkStudentGroup = completeLinkStudentClustNorm.groupby(completeLinkStudentClustNorm['clusters'])


# In[89]:

singleLinkStudentGroup.studentId.groups


# In[90]:

completeLinkStudentGroup.studentId.groups


# In[121]:

#For each cluster, update centroids 
#cdist(studentGroup, cent, 'euclidean'.groups()
completeLinkStudentGroup.count()


# In[122]:

singleLinkStudentGroup.count()


# In[112]:

singleLinkStudentClustNorm.groupby(['clusters']).get_group(1)


# In[94]:

singleLinkStudentClustNorm[['studentId', 'clusters']]


# In[95]:

completeLinkStudentClustNorm[['studentId', 'clusters']]


# In[ ]:




# In[ ]:




# In[96]:

sCluster4 = singleLinkStudentClustNorm[['physics', 'maths', 'english', 'music', 'clusters']].groupby(['clusters']).get_group(4)
sCluster3 = singleLinkStudentClustNorm[['physics', 'maths', 'english', 'music', 'clusters']].groupby(['clusters']).get_group(3)
sCluster2 = singleLinkStudentClustNorm[['physics', 'maths', 'english', 'music', 'clusters']].groupby(['clusters']).get_group(2)
sCluster1 = singleLinkStudentClustNorm[['physics', 'maths', 'english', 'music', 'clusters']].groupby(['clusters']).get_group(1)

del sCluster4['clusters']
del sCluster3['clusters']
del sCluster2['clusters']
del sCluster1['clusters']

slKM4 = KMeans(n_clusters=1).fit(sCluster4)
slKM3 = KMeans(n_clusters=1).fit(sCluster3)
slKM2 = KMeans(n_clusters=1).fit(sCluster2)
slKM1 = KMeans(n_clusters=1).fit(sCluster1)

print("Single Link Cluster Center 4", slKM4.cluster_centers_)
print("Single Link Cluster Center 3", slKM3.cluster_centers_)
print("Single Link Cluster Center 2", slKM2.cluster_centers_)
print("Single Link Cluster Center 1", slKM1.cluster_centers_)


# In[97]:

KM.cluster_centers_


# In[98]:

KM.n_clusters


# In[99]:

cCluster4 = completeLinkStudentClustNorm[['physics', 'maths', 'english', 'music', 'clusters']].groupby(['clusters']).get_group(4)
cCluster3 = completeLinkStudentClustNorm[['physics', 'maths', 'english', 'music', 'clusters']].groupby(['clusters']).get_group(3)
cCluster2 = completeLinkStudentClustNorm[['physics', 'maths', 'english', 'music', 'clusters']].groupby(['clusters']).get_group(2)
cCluster1 = completeLinkStudentClustNorm[['physics', 'maths', 'english', 'music', 'clusters']].groupby(['clusters']).get_group(1)

del cCluster4['clusters']
del cCluster3['clusters']
del cCluster2['clusters']
del cCluster1['clusters']

KM4 = KMeans(n_clusters=1).fit(cCluster4)
KM3 = KMeans(n_clusters=1).fit(cCluster3)
KM2 = KMeans(n_clusters=1).fit(cCluster2)
KM1 = KMeans(n_clusters=1).fit(cCluster1)

print("Complete Link Cluster Center 4", KM4.cluster_centers_)
print("Complete Link Cluster Center 3", KM3.cluster_centers_)
print("Complete Link Cluster Center 2", KM2.cluster_centers_)
print("Complete Link Cluster Center 1", KM1.cluster_centers_)


# In[100]:

Z[:, 3]


# In[101]:

A


# In[123]:

#if 1 & 2 both points are in same cluster, then add 1 to a. If 1 and 2 were in same clusters and now in different clusters,
# then add 1 to b. If 1 and 2 were in different clusters and now in same cluster, then add 1 to c. If both are in different
#clusters the add 1 to d.
a = 0
b = 0
c = 0
d = 0

print("single link length = ", range(len(singleLink)))

for i in range(len(singleLink)):
    print("abcde =", i)
    j = i + 1;
    for j in range(j, len(singleLink)):
        print("j =",j)
        # if 1 and 2 belong to different group and now to the same group then 
        print("singlelink ", singleLink[j] )
        print("completeLink ", completeLink[j] )
        if singleLink[i] == singleLink[j] and completeLink[i] == completeLink[j]:
            a = a + 1
            print ("a =", a)
        if singleLink[i] != singleLink[j] and completeLink[i] != completeLink[j]:
            d = d + 1
            print ("d =", d)
        if singleLink[i] == singleLink[j] and completeLink[i] != completeLink[j]:
            c = c + 1
            print ("c =", c)
        if singleLink[i] != singleLink[j] and completeLink[i] == completeLink[j]:
            b = b + 1
            print ("b =", b)
    
print ("a =", a)
print ("b =", b)
print ("c =", c)
print ("d =", d)

randIndex = (a + d) / (a + b + c + d)
print("randIndex = ", randIndex)


# In[176]:

kmeansClusters = [1, 1, 3, 0, 2, 2, 1, 3, 3, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1,
       3, 1, 1, 0, 3, 2, 0, 2, 2, 0, 0, 2, 0, 1, 0, 1, 0, 1, 1, 0, 0, 3, 2,
       2, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
kmeansClusters.


# In[188]:

kmeansClusters = np.ndarray((69,), buffer=np.array([1, 1, 3, 0, 2, 2, 1, 3, 3, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 0, 3, 2, 0, 2, 2, 0, 0, 2, 0, 1, 0, 1, 0, 1, 1, 0, 0, 3, 2, 2, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
           offset=0,
           dtype=int)


# In[189]:

singleLink


# In[190]:

kmeansClusters


# In[191]:

#if 1 & 2 both points are in same cluster, then add 1 to a. If 1 and 2 were in same clusters and now in different clusters,
# then add 1 to b. If 1 and 2 were in different clusters and now in same cluster, then add 1 to c. If both are in different
#clusters the add 1 to d.
a = 0
b = 0
c = 0
d = 0

for i in range(len(singleLink)):
    print("abcde =", i)
    j = i + 1;
    for j in range(j, len(singleLink)):
        print("j =",j)
        # if 1 and 2 belong to different group and now to the same group then 
        print("singlelink ", singleLink[j] )
        print("kmeansCluster ", kmeansClusters[j] )
        if singleLink[i] == singleLink[j] and kmeansClusters[i] == kmeansClusters[j]:
            a = a + 1
            print ("a =", a)
        if singleLink[i] != singleLink[j] and kmeansClusters[i] != kmeansClusters[j]:
            d = d + 1
            print ("d =", d)
        if singleLink[i] == singleLink[j] and kmeansClusters[i] != kmeansClusters[j]:
            c = c + 1
            print ("c =", c)
        if singleLink[i] != singleLink[j] and kmeansClusters[i] == kmeansClusters[j]:
            b = b + 1
            print ("b =", b)
    
print ("a =", a)
print ("b =", b)
print ("c =", c)
print ("d =", d)

randIndex = (a + d) / (a + b + c + d)
print("randIndex = ", randIndex)


# In[187]:

np.ndarray((69,), buffer=np.array([1, 1, 3, 0, 2, 2, 1, 3, 3, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 0, 3, 2, 0, 2, 2, 0, 0, 2, 0, 1, 0, 1, 0, 1, 1, 0, 0, 3, 2, 2, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
           offset=0,
           dtype=int) # offset = 1*itemsize, i.e. skip first element


# In[ ]:



