import data_loader













if __name__ == "__main__":
    data = data_loader.get_data(file_path, nrows=1000)
    print(f"data set shape is {data.shape}")