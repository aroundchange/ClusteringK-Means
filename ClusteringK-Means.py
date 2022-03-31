# -*- Coding = UTF-8 -*-
# @Time: 2022/3/31 11:44
# @Author: Nico
# File: ClusteringK-Means.py
# @Software: PyCharm


from skimage import io
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('test.jpg')
# io.imshow(image)
# io.show()

rows = image.shape[0]
cols = image.shape[1]

image = image.reshape(image.shape[0] * image.shape[1], 4)
KMeans = KMeans(n_clusters=128, n_init=10, max_iter=200)
KMeans.fit(image)

clusters = np.asarray(KMeans.cluster_centers_, dtype=np.uint8)
labels = np.asarray(KMeans.labels_, dtype=np.uint8)
labels = labels.reshape(rows, cols)

print(clusters.shape)
np.save('test.npy', clusters)
io.imsave('out.jpg', labels)

image = io.imread('test.jpg')
io.imshow(image)
io.show()

