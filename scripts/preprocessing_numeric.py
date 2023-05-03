import pandas as pd
import numpy as np
from scripts import preprocessing_general

def get_columns_ending_with_100g(df):
    """
    Given a DataFrame, returns a list of column names ending with "100g".

    :param df: pandas DataFrame
    :return: list of column names ending with "100g"
    """
    return [col for col in df.columns if col.endswith('_100g')]


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
        #Removes nan by 0, may be worth imputing instead
        df[col].fillna(0, inplace=True)

    return df


def run(df):
    features_list = []
    features_list.extend(get_columns_ending_with_100g(df))

    df = preprocessing_general.drop_columns_not_in_list(df, features_list)

    df = preprocessing_general.remove_columns_with_missing_values(df, 0.7)
    df = clean_100g_columns(df)


    #print(df)
    #print(df.columns)
    return df






