from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd

import matplotlib.pylab as plt

def kMeans(data):
    X = data.drop(columns=['Diabetes_binary'])

    k = 10

    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(X)

    centroids = kmeans.cluster_centers_

    cluster_labels = kmeans.labels_

    data['Cluster'] = cluster_labels

    print("Cluster centroids: ", centroids)
    print('nCluster counts: ', data['Cluster'].value_counts())
    print('Silhouette score: ', silhouette_score(X, cluster_labels))

def doDBSCAN(data):
    features = data.drop(columns=['Diabetes_binary'])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    epsilon = 2
    min_samples = 10

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_features)

    clusters = clusters

    print(type(clusters))

    #print('Cluster labels: ', clusters)
    for i in clusters:
        print(clusters[i])


