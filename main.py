from scripts import data_loader
from scripts import data_explo
from scripts import preprocessing_general
from scripts import preprocessing_numeric
from scripts import preprocessing_textual
from scripts import model_training
from scripts import cluster_interpretation

file_path = "data/en.openfoodfacts.org.products.csv"

if __name__ == "__main__":
    data = data_loader.get_data(file_path, nrows=1000)
    data = preprocessing_general.remove_rows_with_nan(data, "product_name")
    print(f"data set shape is {data.shape}")

