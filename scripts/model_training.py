"""
model_training.py

This module contains functions for training a machine learning model on a preprocessed dataset. It includes functions for:
- Running KMeans clustering on the input DataFrame
- Visualizing the optimal number of clusters using the silhouette score
- Analyzing the results of the clustering
- Running the entire model training pipeline

The main function 'run' executes the model training pipeline on the input DataFrame, including running KMeans clustering with
the optimal number of clusters, analyzing the results, and returning the cluster labels and KMeans model.

Functions:
- run(df)
"""


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def run(df):
    """
    Executes the machine learning model training pipeline on the input DataFrame, including running KMeans clustering with
    the optimal number of clusters, analyzing the results, and returning the cluster labels and KMeans model.

    Args:
        df (pd.DataFrame): The input DataFrame containing preprocessed data.

    Returns:
        tuple: A tuple containing the cluster labels and the KMeans model.
    """
    # Preprocess the data
    # Fill missing values with the column mean
    #df = df.fillna(df.mean())

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Determine the optimal number of clusters using the silhouette score
    silhouette_scores = []
    max_clusters = 10  # You can change this value according to your needs

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))

    optimal_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 because we started from 2 clusters

    # Elbow method visualization
    plt.plot(range(2, max_clusters + 1), silhouette_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Elbow Method')
    plt.show()

    # Apply K-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)

    # Analyze the results
    print("Optimal number of clusters:", optimal_clusters)
    print("Cluster centroids:")
    print(scaler.inverse_transform(kmeans.cluster_centers_))
    print("Data points in each cluster:")
    print(df['cluster'].value_counts())

    # Return both the cluster labels and the KMeans model
    return df['cluster'], kmeans


