import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to load data from an Excel file
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at path {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Function to bin the continuous target variable (e.g., intoxication) into categories
def bin_continuous_target(df, target, n_bins=3):
    binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')  # Uniformly bin into categories
    df[f'{target}_binned'] = binner.fit_transform(df[[target]])
    return df, binner

# Model 3: Predicting binned categories of a continuous measure (e.g., intoxication)
def train_and_evaluate_model3_binned(df):
    features = ['hypertension', 'alertness', 'smoker', 'overweight', 'family_history']
    target = 'intoxication'  # Continuous target measure (intoxication level)
    
    # Bin the target into categories (e.g., low, medium, high intoxication)
    df, binner = bin_continuous_target(df, target, n_bins=3)
    target_binned = f'{target}_binned'
    
    X = df[features]
    y = df[target_binned]  # Predicting the binned categories
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # KNeighborsClassifier for predicting binned categories
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_val_pred = knn.predict(X_val)
    
    # Print classification report
    print("\nClassification Report for Binned Continuous Measure (Intoxication):")
    print(classification_report(y_val, y_val_pred))
    
    # Print accuracy
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {accuracy:.2%}")

# Main script execution
file_path = 'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/test_dataset.xlsx'
df = load_data(file_path)

if df is not None:
    train_and_evaluate_model3_binned(df)
