import pandas as pd
import os

def read_gzipped_csv(file_path):
    """
    Reads a gzipped CSV file and returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path, compression='gzip')
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, 'HotpotQA', 'data', 'dataset_hotpotqa.csv.gz')
    
    dataframe = read_gzipped_csv(file_path)
    if dataframe is not None:
        print(f"Successfully loaded {file_path}")
        output_path = os.path.join(script_dir, 'HotpotQA', 'data', 'dataset_hotpotqa.csv')
        dataframe.to_csv(output_path, index=False)
        print(f"Successfully saved CSV to {output_path}")
        print("First 5 rows of the dataframe:")
        print(dataframe.head()) 