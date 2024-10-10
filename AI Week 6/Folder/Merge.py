import pandas as pd

# Load the two datasets
def load_and_merge_datasets(file1, file2, output_file):
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    
    # Combine both datasets, ignoring duplicate headers
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Save the merged dataset to an Excel file
    combined_df.to_excel(output_file, index=False)
    print(f"Merged dataset saved as {output_file}")

# Merge datasets into one Excel file
file1 = 'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/AI Week 6/Folder/merged_dataset.xlsx'  # Update the path
file2 = 'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/AI Week 6/Folder/set4.xlsx'  # Update the path
output_file = '1000patients.xlsx'
load_and_merge_datasets(file1, file2, output_file)
