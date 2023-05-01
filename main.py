from scripts import data_loader
from scripts import data_explo
from scripts import preprocessing
from scripts import model_training
from scripts import cluster_interpretation

file_path = "data/en.openfoodfacts.org.products.csv"

if __name__ == "__main__":
    data = data_loader.get_data(file_path, nrows=10000)
    data = preprocessing.remove_rows_with_nan(data, "product_name")
    print(f"data set shape is {data.shape}")
# --------------------------------------------------------------------------------------------------------------------------------------------------------
    df = preprocessing.run(data)
#--------------------------------------------------------------------------------------------------------------------------------------------------------
    clusters_labels = model_training.run(df)
    item_names = data["product_name"]
    cluster_interpretation.generate_wordclouds(clusters_labels, df, item_names)
#--------------------------------------------------------------------------------------------------------------------------------------------------------


