import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from the Excel file."""
    try:
        df = pd.read_excel('C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/set2_500_patients.xlsx')
        return df
    except FileNotFoundError:
        print(f"Error: The file at {f'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/set2_500_patients.xlsx'} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        return None

def process_patient_data(df):
    """Process data for all 500 patients, considering 24 hours per day for 30 days."""
    results = []

    for i in range(500):  
        patient_data = df.iloc[i*720:(i+1)*720] 
        results.append({
            'patient_id': i,
            'hypertension': patient_data['hypertension'].any(),
            'family_history': patient_data['family_history'].any(),
            'smoker': patient_data['smoker'].any(),
            'overweight': patient_data['overweight'].any(),
            'intoxication_hours': patient_data['intoxication'].sum(),
            'low_alertness_hours': (patient_data['alertness'] < 0.5).sum(),
            'heart_attack_count': patient_data['heart_attack'].sum(),  
            'accident_count': patient_data['accident'].sum(),  
            'heart_attack_history': patient_data['heart_attack'].sum() > 0,
            'risk_factor_sum': patient_data['hypertension'].any() + 
                               patient_data['family_history'].any() + 
                               patient_data['smoker'].any() + 
                               patient_data['overweight'].any(),
            'died': (patient_data['action'] == 'died').any(),  # Add this line
        })
    
    return pd.DataFrame(results)

def engineer_features(df):
   
    df['hypertension_risk'] = df['hypertension'] * 2  
    df['heart_attack_risk_score'] = df['hypertension_risk'] + df['family_history'] + df['smoker'] + df['overweight'] + df['heart_attack_history'] * 2
    
   
    df['accident_risk_score'] = df['intoxication_hours'] / 24 + df['low_alertness_hours'] / 24 + df['overweight']
    
    return df

def train_and_predict(X, y, X_test):
    """Train a Random Forest model and make predictions."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    val_probas = model.predict_proba(X_val_scaled)[:, 1]
    val_preds = (val_probas > 0.5).astype(int)  # Convert probabilities to binary predictions
    roc_auc = roc_auc_score(y_val, val_probas)
    accuracy = calculate_accuracy(y_val, val_preds)
    print(f"Validation ROC AUC Score: {roc_auc:.4f}")
    print(f"Validation Accuracy: {accuracy:.2%}")
    
    return model.predict_proba(X_test_scaled)[:, 1], roc_auc, accuracy

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy of predictions."""
    return accuracy_score(y_true, y_pred)

def predict_death(heart_attack_probs, accident_probs):
    """Calculate death probability based on heart attack and accident probabilities."""
    return (heart_attack_probs + accident_probs) / 2

def print_patient_predictions(processed_data, heart_attack_probs, accident_probs, death_probs, death_preds, output_file):
    """Write predictions for each patient to a file."""
    with open(output_file, 'w') as f:
        for i, patient in processed_data.iterrows():
            f.write(f"Patient {patient['patient_id']}:\n")
            f.write(f"  Heart Attack Risk: {heart_attack_probs[i]:.2f}%\n")
            f.write(f"  Accident Risk: {accident_probs[i]:.2f}%\n")
            f.write(f"  Death Risk: {death_probs[i]:.2f}%\n")
            f.write(f"  Predicted Death: {'Yes' if death_preds[i] else 'No'}\n")
            f.write(f"  Actual Death: {'Yes' if patient['died'] else 'No'}\n")
            f.write(f"  Actual Heart Attacks: {int(patient['heart_attack_count'])}\n")
            f.write(f"  Actual Accidents: {int(patient['accident_count'])}\n")
            f.write("  Risk Factors:\n")
            f.write(f"    Hypertension History: {'Yes' if patient['hypertension'] else 'No'}\n")
            f.write(f"    Family History: {'Yes' if patient['family_history'] else 'No'}\n")
            f.write(f"    Smoker: {'Yes' if patient['smoker'] else 'No'}\n")
            f.write(f"    Overweight: {'Yes' if patient['overweight'] else 'No'}\n")
            f.write(f"    Hours with Intoxication: {patient['intoxication_hours']:.2f}\n")
            f.write(f"    Hours with Low Alertness: {int(patient['low_alertness_hours'])}\n")
            f.write("\n")

def main():
    file_path = "set3_500_patients.xlsx"  
    output_file = "patient_predictions.txt"  
    data = load_data(file_path)
    if data is not None:
        processed_data = process_patient_data(data)
        processed_data = engineer_features(processed_data)
        
        heart_attack_features = ['hypertension', 'family_history', 'smoker', 'overweight', 'heart_attack_history', 'heart_attack_risk_score']
        accident_features = ['intoxication_hours', 'low_alertness_hours', 'overweight', 'accident_risk_score']
        
        X_heart_attack = processed_data[heart_attack_features]
        X_accident = processed_data[accident_features]
        y_heart_attack = processed_data['heart_attack_count'] > 0  
        y_accident = processed_data['accident_count'] > 0  
        
        print("Training Heart Attack Model:")
        heart_attack_probs, heart_attack_roc_auc, heart_attack_accuracy = train_and_predict(X_heart_attack, y_heart_attack, X_heart_attack)
        heart_attack_probs *= 100  
        
        print("\nTraining Accident Model:")
        accident_probs, accident_roc_auc, accident_accuracy = train_and_predict(X_accident, y_accident, X_accident)
        accident_probs *= 100  
        
        # Predict death probabilities
        death_probs = predict_death(heart_attack_probs, accident_probs)
        death_preds = (death_probs > 50).astype(int)  # Predict death if probability > 50%
        
        # Calculate death prediction accuracy
        death_accuracy = calculate_accuracy(processed_data['died'], death_preds)
        
        print_patient_predictions(processed_data, heart_attack_probs, accident_probs, death_probs, death_preds, output_file)
        print(f"Patient predictions have been written to {output_file}")
        
        print(f"\nHeart Attack Model Accuracy: {heart_attack_accuracy:.2%}")
        print(f"Accident Model Accuracy: {accident_accuracy:.2%}")
        print(f"Death Prediction Accuracy: {death_accuracy:.2%}")
        print(f"Average Death Risk: {death_probs.mean():.2f}%")
        print(f"Patients with Death Risk > 50%: {(death_probs > 50).sum()} out of {len(death_probs)} patients")
        print(f"Actual Deaths: {processed_data['died'].sum()} out of {len(processed_data)} patients")

if __name__ == "__main__":
    main()