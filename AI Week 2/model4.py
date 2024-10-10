import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Function to load data from an Excel file
def load_data(file_path):
    return pd.read_excel(file_path)

# Create sequences of past activities
def create_sequences(df, target_col, n_steps):
    sequences = []
    targets = []
    
    for i in range(n_steps, len(df)):
        seq = df.iloc[i-n_steps:i][target_col].values
        target = df.iloc[i][target_col]
        sequences.append(seq)
        targets.append(target)
    
    return sequences, targets

# Model 4: Forecasting Activity from a Sequence of Past Activities (Naive Bayes)
def train_and_evaluate_model4(df):
    n_steps = 5  # Number of past activities to use for prediction

    # Encode activities as integers
    label_encoder = LabelEncoder()
    df['encoded_action'] = label_encoder.fit_transform(df['action'])

    # Create sequences of past activities for both training and validation sets
    X, y = create_sequences(df, 'encoded_action', n_steps)

    # Convert sequences to DataFrames
    X_df = pd.DataFrame(X)
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X_df, y, test_size=0.2, random_state=42)

    # Naive Bayes Classifier for activity forecasting
    nb_activity_forecast = GaussianNB()
    nb_activity_forecast.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_val_pred = nb_activity_forecast.predict(X_val)
    
    # Decode the predicted activity classes back to original labels
    y_val_pred_decoded = label_encoder.inverse_transform(y_val_pred)
    y_val_decoded = label_encoder.inverse_transform(y_val)
    
    # Print classification report
    print("\nActivity Sequence Forecasting (Naive Bayes) Results:")
    print(classification_report(y_val_decoded, y_val_pred_decoded))

# Main script execution
file_path = 'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/test_dataset.xlsx'
df = load_data(file_path)

if df is not None:
    train_and_evaluate_model4(df)