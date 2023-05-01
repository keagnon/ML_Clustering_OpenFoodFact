import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def run(df):
    # Preprocess the data
    # Fill missing values with the column mean
    #df = df.fillna(df.mean())

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Determine the optimal number of clusters using the silhouette score
    silhouette_scores = []
    max_clusters = 4  # You can change this value according to your needs

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))

    optimal_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 because we started from 2 clusters

    # Apply K-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)

    # Analyze the results
    print("Optimal number of clusters:", optimal_clusters)
    print("Cluster centroids:")
    print(scaler.inverse_transform(kmeans.cluster_centers_))
    print("Data points in each cluster:")
    print(df['cluster'].value_counts())

    return df['cluster']
