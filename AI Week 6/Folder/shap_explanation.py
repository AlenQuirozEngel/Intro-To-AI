import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_excel(file_path)
    df = pd.get_dummies(df, columns=['gender'])
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_columns = [col for col in numerical_columns if not col.startswith('gender_')]
    return df, numerical_columns

def train_model(df, numerical_columns):
    features = numerical_columns + ['gender_female', 'gender_male', 'gender_non-binary']
    target_heart_attack = 'heart_attack'
    
    X = df[features].values
    y_heart_attack = df[target_heart_attack].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y_heart_attack, test_size=0.2, random_state=42)
    
    rf_heart_attack = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_heart_attack.fit(X_train, y_train)
    
    return rf_heart_attack, X_val

def run_shap(model, X_val):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    shap.summary_plot(shap_values[1], X_val)

if __name__ == "__main__":
    file_path = 'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/AI Week 6/Folder/set4_500_patients.xlsx'
    df, numerical_columns = load_data(file_path)
    rf_heart_attack, X_val = train_model(df, numerical_columns)
    run_shap(rf_heart_attack, X_val)