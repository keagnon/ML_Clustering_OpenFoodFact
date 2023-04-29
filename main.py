
"""Data loading tools
"""
import yaml
import pandas as pd
import os
import numpy as np



def read_config(file_path='./config.yaml'):
    """Reads configuration file
    Args:
        file_path (str, optional): file path
    Returns:
        dict: Parsed configuration file
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_data(file_path=None, nrows=None):
    """Loads data
    Args:
        file_path (str, optional): file path of dataset
            By default load data set from static web page
        nrows (int, optional): number or rows to loads from dataset
            By default loads all dataset
    Returns:
        dataframe: output dataframe
    """
    if file_path is None:
        cfg = read_config()
        file_path = cfg['paths']['eng_dataset']
    print("Reading dataset ...")
    return pd.read_csv(file_path,sep="\t", encoding="utf-8",
                       nrows=50, low_memory=False)


if __name__ == "__main__":
    print(os.getcwd())
    data = get_data(file_path = "./data/en.openfoodfacts.org.products.csv", nrows=2000)
    #print(f"data set shape is {data.shape}")
    #print(data.iloc[0])

    




