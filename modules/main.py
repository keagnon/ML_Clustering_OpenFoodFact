import data_loader
import data_exploration
import preprocessing_text
import preprocessing_numerical
import model_training
import cluster_interpretation


file_path = "../data/en.openfoodfacts.org.products.csv"

if __name__ == "__main__":
    data = data_loader.get_data(file_path, nrows=1000)
    print(f"data set shape is {data.shape}")
    data_exploration.types_of_data(data)

