"""
preprocessing_numeric.py

This module contains functions for preprocessing numeric data in pandas DataFrames. It includes functions for:
- Getting columns ending with '_100g'
- Cleaning columns ending with '_100g'
- Running the numeric preprocessing pipeline

The main function 'run' executes the preprocessing pipeline on the input DataFrame, getting the columns ending with '_100g',
dropping columns not in that list, removing columns with more than 70% missing values, and cleaning the '_100g' columns.

Functions:
- get_columns_ending_with_100g(df)
- clean_100g_columns(df)
- run(df)
"""


import pandas as pd
import numpy as np
from scripts import preprocessing_general

def get_columns_ending_with_100g(df):
    """
    Returns a list of column names in a DataFrame that end with '_100g'.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names that end with '_100g'.
    """
    return [col for col in df.columns if col.endswith('_100g')]


def clean_100g_columns(df):
    """
    Cleans columns ending with '_100g' in a given DataFrame by removing values that are not between 0 and 100,
    and replacing NaN values with 0.

    Args:
        df (pd.DataFrame): The input DataFrame containing columns ending with '_100g' to be cleaned.

    Returns:
        pd.DataFrame: A DataFrame with values not between 0 and 100 in the specified columns removed and NaN values replaced with 0.
    """
    columns_100g = [col for col in df.columns if col.endswith('_100g')]

    for col in columns_100g:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(trans_nutrient_value)

        df[col] = df[col].apply(lambda x: x if 0 <= x <= 100 else None)
        #Removes nan by 0, may be worth imputing instead
        df[col].fillna(0, inplace=True)

    return df


def run(df):
    """
    Executes the numeric preprocessing pipeline on the input DataFrame, including getting the columns ending with '_100g',
    dropping columns not in that list, removing columns with more than 70% missing values, and cleaning the '_100g' columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing numeric data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    features_list = []
    features_list.extend(get_columns_ending_with_100g(df))

    df = preprocessing_general.drop_columns_not_in_list(df, features_list)

    df = preprocessing_general.remove_columns_with_missing_values(df, 0.7)
    df = clean_100g_columns(df)


    #print(df)
    #print(df.columns)
    return df






