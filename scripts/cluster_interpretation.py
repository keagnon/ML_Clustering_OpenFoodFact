import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import math

def generate_wordclouds(cluster_labels, data, item_names):
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)

    grid_columns = min(3, n_clusters)
    grid_rows = math.ceil(n_clusters / grid_columns)

    fig, axes = plt.subplots(grid_rows, grid_columns, figsize=(15, 5 * grid_rows))

    for idx, label in enumerate(unique_labels):
        cluster_item_names = item_names[cluster_labels == label]
        text = ' '.join(str(name) for name in cluster_item_names)

        wordcloud = WordCloud(background_color='white', max_words=100).generate(text)

        row_idx = idx // grid_columns
        col_idx = idx % grid_columns

        if grid_rows == 1:
            ax = axes[col_idx]
        else:
            ax = axes[row_idx, col_idx]

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(f"Cluster {label} Word Cloud")

    for idx in range(n_clusters, grid_rows * grid_columns):
        row_idx = idx // grid_columns
        col_idx = idx % grid_columns

        if grid_rows == 1:
            axes[col_idx].axis("off")
        else:
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust the spacing between subplots
    plt.show()
