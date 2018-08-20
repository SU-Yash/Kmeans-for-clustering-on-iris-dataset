#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 14:08:13 2018

@author: suyash
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = pd.read_csv('iris.csv')
iris = np.array(iris)
iris = np.transpose(iris)

model = KMeans(n_clusters = 3)

model.fit(iris)

labels = model.predict(iris)

plt.scatter(iris[:,0],iris[:,1],iris[:,2],c=labels,marker='o')
centroids = model.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1], marker='D')
plt.show()



 