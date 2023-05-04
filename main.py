from scripts import data_loader
from scripts import data_explo
from scripts import preprocessing_general
from scripts import preprocessing_numeric
from scripts import preprocessing_textual
from scripts import model_training
from scripts import cluster_interpretation

file_path = "data/en.openfoodfacts.org.products.csv"


if __name__ == "__main__":
    # Load the first nrows of data from the given file path
    data = data_loader.get_data(file_path, nrows=1000)

    # Remove rows with missing product names from the data
    data = preprocessing_general.remove_rows_with_nan(data, "product_name")

    # Print the shape of the cleaned data
    print(f"data set shape is {data.shape}")

    # Preprocess the data using general preprocessing steps
    df = preprocessing_general.run(data)

    # Get the column names of the preprocessed data (features used)
    Features_used = df.columns

    # Print the features used in the analysis
    print(f'Les features utilisés sont : {Features_used}')

    # Train a k-means clustering model and obtain the cluster labels for each data point
    clusters_labels, kmeans_model = model_training.run(df)

    # Get the product names from the original data
    item_names = data["product_name"]

    # Interpret and display the clusters based on the trained model and the features used
    cluster_interpretation.run(df, clusters_labels, kmeans_model, item_names, Features_used)

