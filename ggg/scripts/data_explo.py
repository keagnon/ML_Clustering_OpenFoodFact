def print_every_column(df):
    for column in df.columns:
        print(column)

def print_example_for_each_column(df, row_number=1):
    for column in df.columns:
        value = df.loc[row_number, column]
        print(value)

def get_unique_values_per_column(df, number_of_value_to_display=3):
    """
    Print for each column a list of X unique values, different from NaN.
    Args:
        df (pd.DataFrame): pandas DataFrame
        number_of_value_to_display (int, optional): Number of unique values to display. Defaults to 3.
    Returns:
        None
    """
    for column in df.columns:
        unique_values = df[column].dropna().unique()
        if len(unique_values) > number_of_value_to_display:
            unique_values = unique_values[:number_of_value_to_display]
        print(f"{column} | {', '.join(map(str, unique_values))}")

def count_nan_in_column(dataframe, column_name):
    """
    Counts the number of NaN values in a specified column of a DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the column to check for NaN values.

    Returns:
        int: The number of NaN values in the specified column.
    """
    print(dataframe[column_name].isna().sum())
