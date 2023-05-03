import pandas as pd
from scripts import preprocessing_general
from scripts import preprocessing_numeric
from scripts import preprocessing_textual
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

def drop_columns_not_in_list(df, columns_to_keep):
    """
    Drop all columns in the DataFrame that are not in the specified list of columns to keep.

    :param df: pandas DataFrame
    :param columns_to_keep: list of column names to keep
    :return: pandas DataFrame with only the specified columns
    """
    return df[columns_to_keep]

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

def split_dataframe(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    textual_columns = data.select_dtypes(include=['object']).columns

    numeric_df = data[numeric_columns]
    textual_df = data[textual_columns]

    return numeric_df, textual_df

def merge_dataframes(numeric_df, textual_df):
    numeric_textual_columns = textual_df.select_dtypes(include=['number']).columns
    return numeric_df.join(textual_df[numeric_textual_columns])


def run(data):
    numeric_df, textual_df = split_dataframe(data)
    numeric_df = preprocessing_numeric.run(numeric_df)
    textual_df = preprocessing_textual.run(textual_df)
    preprocessed_df = merge_dataframes(numeric_df, textual_df)
    return preprocessed_df
    #return numeric_df

if __name__ == "__main__":
    print("Hi")