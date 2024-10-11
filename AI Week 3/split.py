import pandas as pd
from random import shuffle

def samples_per_class(data_y, indices):
    """
    Calculate the number of samples for each class in a specified subset of a dataset.

    Parameters:
    data_y (numpy.ndarray): An array containing class labels for each sample in the dataset.
    indices (list): An array or list of indices specifying which samples to consider for the count.

    Return:
    list: A list of integers where each index corresponds to a class label (0 through 9), and the value at each index
          indicates the number of samples of that class present in the specified subset of `data_y`.
    """
    sample_num = [0 for num in range(10)]
    for idx in indices:
        label = data_y[idx]
        sample_num[label] += 1
    return sample_num

def train_validate_split(data_y, val_ratio=0.2):
    """
    Splits a dataset into training and validation sets based on the specified ratio
    for each class to maintain class distribution balance across both sets.

    Parameters:
    data_y (numpy.ndarray): An array or list containing class labels for each sample in the dataset.
    val_ratio (float, optional): The proportion of the dataset to include in the validation split.
                                 Defaults to 0.2 (20% of the data).

    Returns:
    tuple of lists: A tuple containing two lists:
                     - train_indices (list): Indices of the samples designated for the training set.
                     - val_indices (list): Indices of the samples designated for the validation set.
    """
    sample_num = len(data_y)
    overall_indices = [num for num in range(sample_num)]
    overall_class_num = samples_per_class(data_y, overall_indices)
    val_class_num = [int(num*val_ratio) for num in overall_class_num]
    tmp_val_class_num = [0 for num in range(10)]
    shuffle(overall_indices)
    train_indices = []
    val_indices = []
    for idx in overall_indices:
        tmp_label = data_y[idx]
        if tmp_val_class_num[tmp_label] < val_class_num[tmp_label]:
            val_indices.append(idx)
            tmp_val_class_num[tmp_label] += 1
        else:
            train_indices.append(idx)
    return train_indices, val_indices

# Function to save the split data into two separate CSV files (train.csv and test.csv)
def save_to_csv(data, train_indices, val_indices, file_prefix='MNIST'):
    train_data = data.iloc[train_indices]
    val_data = data.iloc[val_indices]
    
    # Save training data
    train_data.to_csv(f'{file_prefix}_train.csv', index=False)
    
    # Save validation data (used as test data)
    val_data.to_csv(f'{file_prefix}_test.csv', index=False)
    print(f"Training and test data saved as '{file_prefix}_train.csv' and '{file_prefix}_test.csv'.")

# Main logic to load the dataset, split it, and save the train and test data
if __name__ == "__main__":
    # Load the dataset (assumes the dataset is in a CSV file)
    file_path = 'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/Intro-To-AI/AI Week 3/minst.csv'  # Update this to your actual file path
    data = pd.read_csv(file_path)

    # Split data
    labels = data.iloc[:, 0].values  # Assuming labels are in the first column
    train_indices, val_indices = train_validate_split(labels, val_ratio=0.2)

    # Save the split data into CSVs
    save_to_csv(data, train_indices, val_indices)
