from scripts import data_loader
from scripts import preprocessing
from scripts import model_training
from scripts import cluster_interpretation

file_path = "data/en.openfoodfacts.org.products.csv"




if __name__ == "__main__":
    data = data_loader.get_data(file_path, nrows=50)
    print(f"data set shape is {data.shape}")
