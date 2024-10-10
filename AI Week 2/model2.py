import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Function to load data from an Excel file
def load_data(file_path):
    """Load the dataset from the provided file path."""
    try:
        return pd.read_excel(file_path)  # Use pd.read_csv() if it's a CSV
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Model 2: Prediction of the Activity Class
def train_and_evaluate_model2(df):
    """Train a model to predict the activity class."""
    features = ['hypertension', 'alertness', 'intoxication', 'smoker', 'overweight', 'family_history']
    target_activity = 'action'  # Assuming 'action' column represents activity class
    
    # Ensure the necessary columns exist
    if not all(col in df.columns for col in features + [target_activity]):
        print("Error: Required columns are missing from the dataset.")
        return

    X = df[features]
    y_activity = df[target_activity]
    
    # Train/test split
    X_train, X_val, y_train_activity, y_val_activity = train_test_split(X, y_activity, test_size=0.2, random_state=42)
    
    # Train RandomForestClassifier for activity prediction
    rf_activity = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_activity.fit(X_train, y_train_activity)
    
    # Make predictions
    y_val_pred_activity = rf_activity.predict(X_val)
    
    # Print results
    print("\nActivity Class Prediction Results:")
    print(classification_report(y_val_activity, y_val_pred_activity))

# Main script execution
file_path = 'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/test_dataset.xlsx'
df = load_data(file_path)

if df is not None:
    train_and_evaluate_model2(df)
