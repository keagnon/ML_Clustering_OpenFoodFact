import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from wordcloud import WordCloud
import matplotlib.pyplot as plt



# Add the modified generate_wordclouds function
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
