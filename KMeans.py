#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 14:08:13 2018

@author: suyash sardar
"""

from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = pd.read_csv('iris.csv') # Importing Data Set
iris = np.array(iris)
iris = np.transpose(iris)

model = KMeans(n_clusters = 3)  # Defining the KMeans Model, 

model.fit(iris)  # Fitting the model on the data. It'll group the data in 3 clusters

labels = model.predict(iris)  # Predicting the cluster label for data based on the model 

plt.scatter(iris[:,0],iris[:,1],iris[:,2],c=labels,marker='o')
centroids = model.cluster_centers_

plt.scatter(centroids[:,0],centroids[:,1], marker='D') # Visulizing Results
plt.show()



 
