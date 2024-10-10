from model import load_data, train_and_evaluate_model
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

def get_model_statistics():
    file_path = 'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/AI Week 6/Folder/set4_500_patients.xlsx'
    df, numerical_columns = load_data(file_path)
    
    rf_accident, rf_heart_attack, X_val, y_val_accident, y_val_heart, gender, features = train_and_evaluate_model(df, numerical_columns)
    
    # Get predictions
    y_pred_accident = rf_accident.predict(X_val)
    y_pred_heart_attack = rf_heart_attack.predict(X_val)
    
    # Calculate accuracy
    accident_accuracy = accuracy_score(y_val_accident, y_pred_accident)
    heart_attack_accuracy = accuracy_score(y_val_heart, y_pred_heart_attack)
    
    # Generate classification reports
    accident_report = classification_report(y_val_accident, y_pred_accident)
    heart_attack_report = classification_report(y_val_heart, y_pred_heart_attack)
    
    return {
        'accident_accuracy': accident_accuracy,
        'heart_attack_accuracy': heart_attack_accuracy,
        'accident_report': accident_report,
        'heart_attack_report': heart_attack_report
    }

if __name__ == "__main__":
    stats = get_model_statistics()
    print("Accident Prediction Accuracy:", stats['accident_accuracy'])
    print("\nAccident Classification Report:")
    print(stats['accident_report'])
    print("\nHeart Attack Prediction Accuracy:", stats['heart_attack_accuracy'])
    print("\nHeart Attack Classification Report:")
    print(stats['heart_attack_report'])
