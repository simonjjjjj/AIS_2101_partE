from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pylab as plt

def kMeans(data):
    X = data.drop(columns=['Diabetes_binary'])

    # hyper-parameter
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

    # hyper-parameters
    epsilon = 3
    min_samples = 50

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_features)

    dbscanHistogram(clusters)   # visualizing clusters with histogram



def dbscanHistogram(labels):
    # Printing number of noise points
    num_noise = np.sum(labels == -1)
    print(num_noise)

    # Count the number of data points in each cluster (excluding noise points)
    unique_labels, cluster_counts = np.unique(labels[labels != -1], return_counts=True)
    print(unique_labels)

    # Create a bar plot to visualize the cluster sizes
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, cluster_counts, color='skyblue')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Data Points')
    plt.title('Histogram of Cluster Sizes')
    plt.xticks(unique_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()