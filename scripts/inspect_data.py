import pandas as pd

file_path = 'data/PCA_CICIDS2017.csv'

try:
    # Read only the first 10 columns to avoid memory issues
    df = pd.read_csv(file_path, usecols=range(10))
    print("Successfully loaded the first 10 columns.")
    print("Columns:", df.columns)
    print("\nInfo:")
    df.info()
    print("\nHead:")
    print(df.head())
except Exception as e:
    print(f"An error occurred: {e}")
