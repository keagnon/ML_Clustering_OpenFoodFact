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

def null_per_100g_column(df, treshold=70):
    col_names = []
    for x in df.columns:
        col_names.append(x)

    all_nutrition_cols = [x for x in col_names if '_100g' in x]
    print('Pourcentage de valeurs Null par colonnes :\n')

    i = 0

    nutrition_col_to_keep = []

    for col in all_nutrition_cols:
        res = (df[col].isnull().sum() / len(df)) * 100
        res = round(res, 2)
        if res < treshold:
            nutrition_col_to_keep.append(col)
            i += 1
            print(f'   - {col} : {res}%')


    print(f"\nNombre colonnes avec moins de {treshold}% de valeurs manquantes: ", i)



def types_of_data(df):
    data_types_dict = {}
    type_counts = df.dtypes.value_counts()
    print("Data types and their counts:")
    for dtype, count in type_counts.items():
        print(f"{dtype}: {count}")
    for col_name, col_type in df.dtypes.items():
        if col_type in data_types_dict:
            data_types_dict[col_type].append(col_name)
        else:
            data_types_dict[col_type] = [col_name]
    return data_types_dict


if __name__ == "__main__":
    print(__name__)
