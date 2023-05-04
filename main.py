from scripts import data_loader
from scripts import data_explo
from scripts import preprocessing
from scripts import model_training
from scripts import cluster_interpretation

file_path = "data/en.openfoodfacts.org.products.csv"


if __name__ == "__main__":
    data = data_loader.get_data(file_path, nrows=1000)
    data = preprocessing_general.remove_rows_with_nan(data, "product_name")
    print(f"data set shape is {data.shape}")

    df = preprocessing_general.run(data)
    Features_used = df.columns
    print(f'Les features utilisés sont : {Features_used}')

    clusters_labels, kmeans_model = model_training.run(df)
    item_names = data["product_name"]
    cluster_interpretation.generate_wordclouds(clusters_labels, df, item_names)
#--------------------------------------------------------------------------------------------------------------------------------------------------------

    cluster_interpretation.run(df, clusters_labels, kmeans_model, item_names, Features_used)

