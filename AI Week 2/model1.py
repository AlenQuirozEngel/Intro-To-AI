import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
def load_data(file_path):
    return pd.read_excel(file_path)

# Model 1: Prediction of Accidents and Heart Attacks
def train_and_evaluate_model1(df):
    features = ['hypertension', 'alertness', 'intoxication', 'smoker', 'overweight', 'family_history']
    target_accident = 'accident'
    target_heart_attack = 'heart_attack'
    
    # Train/test split
    X = df[features]
    y_accident = df[target_accident]
    y_heart_attack = df[target_heart_attack]
    
    # Split into training and validation sets
    X_train, X_val, y_train_accident, y_val_accident = train_test_split(X, y_accident, test_size=0.2, random_state=42)
    X_train_heart, X_val_heart, y_train_heart, y_val_heart = train_test_split(X, y_heart_attack, test_size=0.2, random_state=42)
    
    # Train RandomForestClassifier for accidents and heart attacks
    rf_accident = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_heart_attack = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rf_accident.fit(X_train, y_train_accident)
    rf_heart_attack.fit(X_train_heart, y_train_heart)
    
    # Make predictions
    y_val_pred_accident = rf_accident.predict(X_val)
    y_val_pred_heart_attack = rf_heart_attack.predict(X_val_heart)
    
    # Print results
    print("\nAccident Prediction Results:")
    print(classification_report(y_val_accident, y_val_pred_accident))
    
    print("\nHeart Attack Prediction Results:")
    print(classification_report(y_val_heart, y_val_pred_heart_attack))

# Load the dataset and train Model 1
file_path = 'C:/Users/alenq\/Documents/Computer_Science_Course_UM/repository year 2/test_dataset.xlsx'
df = load_data(file_path)
train_and_evaluate_model1(df)
