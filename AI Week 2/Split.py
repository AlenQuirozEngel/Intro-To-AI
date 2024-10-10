import pandas as pd

# Function to split the dataset into training, validation, and testing sets
def split_dataset(df):
    train_df = df[df['subject'] < 350]   # First 350 subjects for training
    val_df = df[(df['subject'] >= 350) & (df['subject'] < 450)]  # Next 100 subjects for validation
    test_df = df[df['subject'] >= 450]  # Last 50 subjects for testing
    
    return train_df, val_df, test_df

# Function to save the split datasets into Excel files
def save_datasets_to_excel(train_df, val_df, test_df, output_train, output_val, output_test):
    train_df.to_excel(output_train, index=False)
    val_df.to_excel(output_val, index=False)
    test_df.to_excel(output_test, index=False)
    print(f"Training set saved as {output_train}")
    print(f"Validation set saved as {output_val}")
    print(f"Testing set saved as {output_test}")

# Example usage: load the merged dataset and split it
merged_file = 'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/merged_dataset.xlsx'  # Replace with actual merged dataset path
merged_df = pd.read_excel(merged_file)

# Split the merged dataset
train_df, val_df, test_df = split_dataset(merged_df)

# Save the split datasets to Excel files
output_train = 'train_dataset.xlsx'
output_val = 'validation_dataset.xlsx'
output_test = 'test_dataset.xlsx'

save_datasets_to_excel(train_df, val_df, test_df, output_train, output_val, output_test)
