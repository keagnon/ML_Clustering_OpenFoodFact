import numpy as np
import pandas as pd
import data_loader
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def calculate_centroids(cluster_labels, data):
    """Calculates centroids of clusters in the dataset.

    Args:
        cluster_labels (list): List of cluster labels for each data point.
        data (pd.DataFrame): Dataframe containing the dataset.

    Returns:
        pd.DataFrame: Centroids of clusters.
    """
    centroids = []
    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
        cluster_data = data[cluster_labels == label]
        centroid = cluster_data.mean(axis=0)
        centroids.append(centroid)

    return pd.DataFrame(centroids)


def get_most_common_features(cluster_labels, data, num_features=5):
    """Finds the most common features in each cluster.

    Args:
        cluster_labels (list): List of cluster labels for each data point.
        data (pd.DataFrame): Dataframe containing the dataset.
        num_features (int, optional): Number of top features to return.

    Returns:
        list: List of dictionaries containing the most common features.
    """
    most_common_features = []
    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
        cluster_data = data[cluster_labels == label]
        feature_counts = Counter()

        for _, row in cluster_data.iterrows():
            feature_counts.update(row)

        most_common = dict(feature_counts.most_common(num_features))
        most_common_features.append(most_common)

    return most_common_features


def summarize_clusters(cluster_labels, data, centroids, most_common_features):
    """Prints a summary of the clusters.

    Args:
        cluster_labels (list): List of cluster labels for each data point.
        data (pd.DataFrame): Dataframe containing the dataset.
        centroids (pd.DataFrame): Centroids of clusters.
        most_common_features (list): List of dictionaries containing the most common features.
    """
    unique_labels = np.unique(cluster_labels)
    print(f"There are {len(unique_labels)} clusters.")

    for idx, label in enumerate(unique_labels):
        cluster_size = len(data[cluster_labels == label])
        print(f"Cluster {label} has {cluster_size} members.")
        print(f"Centroid of cluster {label}:")
        print(centroids.loc[idx])
        print(f"Most common features in cluster {label}:")
        print(most_common_features[idx])
        print()


def select_numeric_columns(data):
    """Selects only numeric columns from the dataset.

    Args:
        data (pd.DataFrame): Dataframe containing the dataset.

    Returns:
        pd.DataFrame: Dataframe with only numeric columns.
    """
    numeric_columns = data.select_dtypes(include=np.number)
    return numeric_columns



def generate_wordclouds(cluster_labels, data, item_names):
    """Generates word clouds for each cluster based on item names.

    Args:
        cluster_labels (list): List of cluster labels for each data point.
        data (pd.DataFrame): Dataframe containing the dataset.
        item_names (pd.Series): Series containing item names.
    """
    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
        cluster_item_names = item_names[cluster_labels == label]
        text = ' '.join(str(name) for name in cluster_item_names)  # Convert item names to strings before joining

        wordcloud = WordCloud(background_color='white', max_words=100).generate(text)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Cluster {label} Word Cloud")
        plt.show()



def impute_missing_values(data):
    """
    Impute missing values in a DataFrame using the mean of the respective column.

    :param data: A pandas DataFrame with missing values.
    :return: A pandas DataFrame with missing values imputed.
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Apply the imputer only to numeric columns with any data
    imputed_data = data.copy()
    for col in data.columns:
        if data[col].dtype.kind in 'biufc' and not data[col].isna().all():
            imputed_data[col] = imputer.fit_transform(data[col].to_numpy().reshape(-1, 1)).flatten()

    return imputed_data

def display_centroids(centroids_df):
    """
    Display the centroids in a tabular format for better readability.

    :param centroids_df: A DataFrame representing the centroids
    """
    pd.set_option('display.float_format', '{:.3f}'.format)

    # Transpose the DataFrame, so each column represents a cluster and each row a feature
    centroids_df = centroids_df.T

    # Round the values to 2 decimal places
    centroids_df = centroids_df.round(3)

    # Reset the index, and name the index and columns
    centroids_df.reset_index(inplace=True)
    centroids_df.index.name = "Feature"
    centroids_df.columns = ["Feature"] + [f"Cluster {i}" for i in range(len(centroids_df.columns) - 1)]

    # Set the index to the feature names
    centroids_df.set_index("Feature", inplace=True)

    # Use pandas to print the DataFrame
    print(centroids_df)

if __name__ == "__main__":
    """
    This script loads the dataset, preprocesses it by selecting numeric columns,
    imputing missing values, and scaling the data. Then, it applies KMeans clustering
    to find 4 clusters and summarizes their characteristics.
    """
    data = data_loader.get_data("..\data\en.openfoodfacts.org.products.csv", 100)
    #Should drop columns that are not meaningful
    numeric_data = select_numeric_columns(data)

    imputed_data = impute_missing_values(numeric_data)
    imputed_data = imputed_data.dropna(axis=1, how='all')

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(imputed_data)
    kmeans = KMeans(n_clusters=4)
    cluster_labels = kmeans.fit_predict(scaled_data)

    centroids = calculate_centroids(cluster_labels, imputed_data)
    display_centroids(centroids)
    #most_common_features = get_most_common_features(cluster_labels, imputed_data)
    #summarize_clusters(cluster_labels, imputed_data, centroids, most_common_features)

    # Assuming the item names are stored in a column called "product_name"
    item_names = data["product_name"]

    # Call the generate_wordclouds function
    generate_wordclouds(cluster_labels, imputed_data, item_names)
