
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest



def null__rate_features(df, tx_threshold=50):
    """
        ------------------------------------------------------------------------------
        Goal : 
            - Calculate the null rate of each variable in a dataframe
        ------------------------------------------------------------------------------
        Parameters:
        - df : 
            Dataset to be analyzed
        - tx_threshold : 
            nullity threshold above which a variable is considered to have a high 
            nullity rate (by default 50%)
        -----------------------------------------------------------------------------
        Return :
        - high_null_rate :
            A list of variables with their null rate greater than or equal 
            to the specified threshold,sorted in descending order of null rate
        -----------------------------------------------------------------------------
    """
    
    #Calcul of null rate of each variable
    null_rate = ((df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    null_rate.columns = ['Variable','Taux_de_Null']
    
    #Selection of variables with a null rate greater than or equal to the specified threshold
    high_null_rate = null_rate[null_rate.Taux_de_Null >= tx_threshold]
    
    return high_null_rate

def fill_rate_features(df):
    
    """
        ---------------------------------------------------------------------------------
        Goal : 
            - Calculate the fill rate of each variable in a DataFrame
            - Displays a horizontal bar graph that shows the fill rate for each variable.
        ---------------------------------------------------------------------------------
        Parameters:
            - df : Dataframe to be analyzed
        ---------------------------------------------------------------------------------
    """
    
    #Calculate the rate of nullity for each variable 
    filling_features = null__rate_features(df, 0)
    
    #Calculate of fill rate by subtracting the null rate from 100%
    filling_features["fill_rate"] = 100-filling_features["Taux_de_Null"]
    
    #Sort results in descending order of filling rate
    filling_features = filling_features.sort_values("Taux_de_Null", ascending=False)
    
    # Creating the horizontal bar chart with Seaborn
    fig = plt.figure(figsize=(20, 35))
    font_title = {'family': 'serif',
                'color':  '#114b98',
                'weight': 'bold',
                'size': 18, 
                }
    sns.barplot(x="fill_rate", y="Variable", data=filling_features, palette="flare")
    plt.axvline(linewidth=2, color = 'r')
    plt.title("Taux de remplissage des variables dans le jeu de données (%)", fontdict=font_title)
    plt.xlabel("Taux de remplissage (%)")
    
    return plt.show()



def clean_dataframe(df,colonnes_a_garder):

    #Delete all columns except those to keep
    colonnes_a_supprimer = [col for col in df.columns if col not in colonnes_a_garder]
    df_garder = df.drop(colonnes_a_supprimer, axis=1)

    return df_garder


def select_nutrition_col(data,all_nutrition_cols):
    #print('Pourcentage de valeurs Null par colonnes :\n')

    i=0

    nutrition_col_to_keep = []

    for col in all_nutrition_cols:
        res = (data[col].isnull().sum() / len(data)) * 100
        res = round(res,2)
        if res < 50:
            nutrition_col_to_keep.append(col)
            i += 1
            #print(f'   - {col} : {res}%')
            
    #print("\nNombre colonnes : " , i)
    return nutrition_col_to_keep

def drop_duplicates(df):
    df = df.drop_duplicates()
    return df


def remove_inconsistent_values(df):
    # List of columns with type float
    colonnes_float = df.select_dtypes(include=['float']).columns
    
    # Remove inconsistent values
    for col in colonnes_float:
        # Replace negative values with NaN
        df.loc[df[col] < 0, col] = np.nan
        
    return df


def clean_filter_dataframe(final_df):

    """
        ---------------------------------------------------------------------------------
        Goal : 
            - Clean dataset
        ---------------------------------------------------------------------------------
        Parameters:
            - final_df : Dataframe to be analyzed
        ---------------------------------------------------------------------------------
    """

    drop_duplicates(final_df)

    #### Delete negative value from dataframe
    remove_inconsistent_values(final_df)

    #### Remove any leading or trailing spaces from column names
    final_df.columns = final_df.columns.str.strip()

    #### Convert all string columns to lowercase
    string_cols = final_df.select_dtypes(include='object').columns
    final_df[string_cols] = final_df[string_cols].apply(lambda x: x.str.lower())

    # Convert any date columns to datetime format
    date_cols = final_df.select_dtypes(include='datetime').columns
    final_df[date_cols] = final_df[date_cols].apply(pd.to_datetime)

    # Remove any non-numeric characters from numeric columns
    numeric_cols = final_df.select_dtypes(include='number').columns
    final_df[numeric_cols] = final_df[numeric_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('[^0-9\.]+', ''), errors='coerce'))

    #select columns with type float and int
    final_df.select_dtypes(include=['float', 'int'])
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()

    #select columns with type object
    object_cols = final_df.select_dtypes(include=['object']).columns.tolist()

    #Input value base on the strategy choosen for numerical columns
    """strategy='median', missing_values=np.nan"""
    imputer = SimpleImputer(strategy='constant',
                            missing_values=np.nan, fill_value=0)

    imputer = imputer.fit(final_df[numeric_cols])
    final_df[numeric_cols] = imputer.transform(final_df[numeric_cols])

    #Input value base on the strategy choosen for object columns
    imputer_categoriel = SimpleImputer(strategy='most_frequent', 
                            missing_values=np.nan)
    imputer_categoriel = imputer_categoriel.fit(final_df[object_cols])
    final_df[object_cols] = imputer_categoriel.transform(final_df[object_cols])
    
    return final_df


