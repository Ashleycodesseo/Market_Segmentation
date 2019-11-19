#%%
#Market Segmentation 
#Goal segment the customer data of a retail shop based on
#their satisfaction and brand loyalty
#Satisfaction was taken from a 1-10 customer survey
#Brand loyalty was based on the number of customer purchases
#made over the course of a year.
#%%
#Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
from sklearn.cluster import KMeans 
#%%
#Load the Data
data = pd.read_csv('3.12. Example.csv')
print(data)
#%%
#Plotting the Data
plt.scatter(data['Satisfaction'], data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

#%%
#Checkpoint
x= data.copy()
#%%
#Standardize the Variables
#The values of the Satisfaction input are not standardized
#we need to fix it so Loyalty and Satisfaction have equal weight 
#for our clusters
from sklearn import preprocessing

x_scaled = preprocessing.scale(x)

print(x_scaled)
#%%
#Finding the Right Number of Clusters Using the Elbow Method

wcss=[]

for i in range(1,10):
    
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
print(wcss)

#%%
# Plot the number of clusters vs WCSS
plt.plot(range(1,10), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
#Now given the elbow, we should decide how many clusters(2,3,4, or 5)

#%%
#Exploring the Clusters
kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new= x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)
print(display(clusters_new))

#so in this new table, the data contains the original values
#but has the predictions based on the standardized values
#we will plot the data without standardizing the axis, but the
#solution will be the standardized one
#%%
#Plotting the Clusters
plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
#%%
#Interpreting the results:
# From the plot of the graph, we have found the best fit of four customer types
# that explains their satisfaction and likeliness to purchase often. They are:
# Alienated: Located in the lower left quadrant with Low Satisfaction/Low Brand Loyalty
# Supporters: Located in the Upper Left Quadrant, frequent the merchant often but have middling to Low Satisfaction
#Fans: The Upper Right quadrant are both loyal and deeply satisfied with the brand
#Roamers: Lower right quadrant, where they are satisfied with the service but not so interested in buying often
#%%
#What's Next?
#Based on this information, it would be important to gather more data 
#from customers to see where they fall on they fall on the graph and what would
#make them feel happier with the service enough to return shopping again.