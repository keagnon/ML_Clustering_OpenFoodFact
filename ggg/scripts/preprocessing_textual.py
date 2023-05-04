"""
preprocessing_textual.py

This module contains functions for preprocessing textual data in pandas DataFrames. It includes functions for:
- Encoding categorical data
- Processing the 'nutriscore_grade' column
- Processing the 'main_category_en' column
- Processing the 'additives_tags' column
- Running the textual preprocessing pipeline

The main function 'run' executes the preprocessing pipeline on the input DataFrame, applying the necessary functions depending on the presence of specific columns in the DataFrame.

Functions:
- encode_data(data, column_name)
- process_nutriscore_grade(data)
- process_main_category_en(data)
- process_additives_tags(data)
- run(data)
"""


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def encode_data(data, column_name):
    """
    Encodes a categorical column using the LabelEncoder.

    Args:
        data (pd.DataFrame): The input DataFrame containing the categorical column.
        column_name (str): The name of the column to be encoded.

    Returns:
        pd.DataFrame: A DataFrame with the encoded column added as '{column_name}_encoded'.
    """
    data = data.copy()
    le = LabelEncoder()
    data.loc[:, f'{column_name}_encoded'] = le.fit_transform(data[column_name].astype(str))
    return data


def process_nutriscore_grade(data):
    """
    Encodes the 'nutriscore_grade' column in the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing the 'nutriscore_grade' column.

    Returns:
        pd.DataFrame: A DataFrame with the encoded 'nutriscore_grade' column added as 'nutriscore_grade_encoded'.
    """
    return encode_data(data, 'nutriscore_grade')


def process_main_category_en(data):
    """
    Encodes the 'main_category_en' column in the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing the 'main_category_en' column.

    Returns:
        pd.DataFrame: A DataFrame with the encoded 'main_category_en' column added as 'main_category_en_encoded'.
    """
    return encode_data(data, 'main_category_en')


def process_additives_tags(data):
    """
    Processes the 'additives_tags' column in the input DataFrame by removing the part before the colon in the tags,
    replacing NaN values with "No additives", and counting the number of additives for each product.

    Args:
        data (pd.DataFrame): The input DataFrame containing the 'additives_tags' column.

    Returns:
        pd.DataFrame: A DataFrame with the processed 'additives_tags' column and a new 'additives_count' column.
    """
    data = data.copy()

    # Remove the part before the colon in the tags
    data['additives_tags'] = data['additives_tags'].apply(
        lambda x: ','.join(tag.split(':')[-1] for tag in str(x).split(',')))

    # Replace NaN values with "No additives"
    data['additives_tags'] = data['additives_tags'].fillna('No additives')

    # Count the number of additives for each product
    data['additives_count'] = data['additives_tags'].apply(lambda x: 0 if x == 'No additives' else len(x.split(',')))

    return data.fillna(0)




def run(data):
    """
    Executes the textual preprocessing pipeline on the input DataFrame, applying the necessary functions
    depending on the presence of specific columns in the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing textual data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    if 'nutriscore_grade' in data.columns:
        data = process_nutriscore_grade(data)
    if 'main_category_en' in data.columns:
        data = process_main_category_en(data)
    if 'additives_tags' in data.columns:
        data = process_additives_tags(data)
    return data
