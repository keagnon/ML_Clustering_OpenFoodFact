import pandas as pd
import numpy as np


def get_columns_ending_with_100g(df):
    """
    Given a DataFrame, returns a list of column names ending with "100g".

    :param df: pandas DataFrame
    :return: list of column names ending with "100g"
    """
    return [col for col in df.columns if col.endswith('_100g')]

def remove_rows_with_nan(dataframe, column_name):
    """
    Removes rows with NaN values in the specified column from the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the column to check for NaN values.

    Returns:
        pd.DataFrame: A new DataFrame with rows containing NaN values in the specified column removed.
    """
    return dataframe.dropna(subset=[column_name])

def clean_100g_columns(df):
    """
    This function removes values that are not between 0 and 100 in any columns ending with '_100g' in a given DataFrame
    and replaces NaN values with 0.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing columns ending with '_100g' to be cleaned.

    Returns:
    pandas.DataFrame: A DataFrame with values not between 0 and 100 in the specified columns removed and NaN values replaced with 0.
    """
    columns_100g = [col for col in df.columns if col.endswith('_100g')]

    for col in columns_100g:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(trans_nutrient_value)

        df[col] = df[col].apply(lambda x: x if 0 <= x <= 100 else None)
        df[col].fillna(0, inplace=True)

    return df


def remove_columns_with_missing_values(df, threshold=0.8):
    """
    Remove columns with more than a given percentage of missing values from a DataFrame.

    :param df: pandas DataFrame
    :param threshold: float, percentage of missing values to remove the column (default: 0.8)
    :return: pandas DataFrame with the specified columns removed
    """
    missing_values_ratio = df.isnull().sum() / len(df)
    columns_to_keep = missing_values_ratio[missing_values_ratio <= threshold].index

    return df[columns_to_keep]

def drop_columns_not_in_list(df, columns_to_keep):
    """
    Drop all columns in the DataFrame that are not in the specified list of columns to keep.

    :param df: pandas DataFrame
    :param columns_to_keep: list of column names to keep
    :return: pandas DataFrame with only the specified columns
    """
    return df[columns_to_keep]

def run(df):
    features_list = []
    features_list.extend(get_columns_ending_with_100g(df))

    df = drop_columns_not_in_list(df, features_list)

    df = remove_columns_with_missing_values(df, 0.9)
    df = clean_100g_columns(df)


    #print(df)
    #print(df.columns)
    return df






