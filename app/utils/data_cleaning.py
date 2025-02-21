# app/utils/data_cleaning.py

import pandas as pd

def remove_outliers(df, column):
    """
    Remove outliers from a specified column using the IQR method.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame.
    - column (str): Column name to remove outliers from.

    Returns:
    - pandas.DataFrame: DataFrame without outliers in the specified column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_clean

def clean_stock_data(file_path, processed_path, columns_to_clean):
    """
    Load stock data, remove outliers from specified columns, and save cleaned data.

    Parameters:
    - file_path (str): Path to the raw stock data CSV file.
    - processed_path (str): Path to save the cleaned stock data CSV file.
    - columns_to_clean (list): List of column names to clean.

    Returns:
    - pandas.DataFrame: Cleaned stock data.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    print("Loaded stock data")

    for col in columns_to_clean:
        df_before = df.shape[0]
        df = remove_outliers(df, col)
        df_after = df.shape[0]
        print(f"Removed {df_before - df_after} outliers from column '{col}'")

    df.to_csv(processed_path, index=False)
    print(f"Cleaned stock data saved to '{processed_path}'")
    return df