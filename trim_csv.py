import pandas as pd
import os

def trim_csv_in_place(filepath, rows=10000):
    # Get the file name from the path
    file_name = os.path.basename(filepath)
    
    # Confirm before proceeding
    confirm = input(f"This will trim {file_name} to {rows} rows. Proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    # Read the first 10,000 rows of the CSV file
    df = pd.read_csv(filepath, nrows=rows)
    
    # Overwrite the original file with the trimmed data
    df.to_csv(filepath, index=False)
    
    print(f"CSV file {file_name} has been trimmed to {rows} rows.")

# File path
file_path = "/Users/skylarshi/Documents/Streamlit/Personality_test.csv"
trim_csv_in_place(file_path)