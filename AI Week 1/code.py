import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def process_patient_data(df):
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
            'died': patient_data['action'].eq('died').any(),
        })
    return pd.DataFrame(results)

def engineer_features(df):
    df['hypertension_risk'] = df['hypertension'] * 2
    df['heart_attack_risk_score'] = df['hypertension_risk'] + df['family_history'] + df['smoker'] + df['overweight'] + df['heart_attack_history'] * 2
    df['accident_risk_score'] = (df['intoxication_hours'] / 24 + 
                                 df['low_alertness_hours'] / 24 + 
                                 df['overweight'])
    return df

def rule_based_prediction(df):
    df['pred_heart_attack'] = (df['hypertension'] & df['smoker'] & df['overweight']) | (df['heart_attack_history'])
    df['pred_accident'] = (df['intoxication_hours'] > 12) | (df['low_alertness_hours'] > 12)
    df['pred_death'] = df['pred_heart_attack'] | df['pred_accident'] | (df['heart_attack_count'] > 0)
    return df

def train_and_predict(X, y, X_test):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, min_samples_split=10, min_samples_leaf=5)
    model.fit(X_train_scaled, y_train)
    
    val_probas = model.predict_proba(X_val_scaled)[:, 1]
    val_preds = (val_probas > 0.5).astype(int)
    roc_auc = roc_auc_score(y_val, val_probas)
    accuracy = accuracy_score(y_val, val_preds)
    
    print(f"Validation ROC AUC Score: {roc_auc:.4f}")
    print(f"Validation Accuracy: {accuracy:.2%}")
    
    return model.predict_proba(X_test_scaled)[:, 1], roc_auc, accuracy

def print_predictions_to_terminal_and_file(processed_data, heart_attack_probs, accident_probs, death_probs, death_preds):
    with open('predictions.txt', 'w') as f:
        for i, patient in processed_data.iterrows():
            patient_info = (
                f"Patient {patient['patient_id']}:\n"
                f"  Heart Attack Risk: {heart_attack_probs[i]:.2f}%\n"
                f"  Accident Risk: {accident_probs[i]:.2f}%\n"
                f"  Death Risk: {death_probs[i]:.2f}%\n"
                f"  Predicted Death: {'Yes' if death_preds[i] else 'No'}\n"
                f"  Actual Death: {'Yes' if patient['died'] else 'No'}\n"
                f"  Actual Heart Attacks: {int(patient['heart_attack_count'])}\n"
                f"  Actual Accidents: {int(patient['accident_count'])}\n"
                f"    Hypertension History: {'Yes' if patient['hypertension'] else 'No'}\n"
                f"    Family History: {'Yes' if patient['family_history'] else 'No'}\n"
                f"    Smoker: {'Yes' if patient['smoker'] else 'No'}\n"
                f"    Overweight: {'Yes' if patient['overweight'] else 'No'}\n"
                f"    Hours with Intoxication: {patient['intoxication_hours']:.2f}\n"
                f"    Hours with Low Alertness: {int(patient['low_alertness_hours'])}\n\n"
            )
            
            # Write to file
            f.write(patient_info)
            
            # Also print to terminal
            print(patient_info)

def print_confusion_matrices_and_accuracy(processed_data, heart_attack_probs, accident_probs, death_probs, death_preds):
    # Calculate accuracies
    heart_attack_ml_acc = accuracy_score(processed_data['heart_attack_count'] > 0, heart_attack_probs > 50)
    accident_ml_acc = accuracy_score(processed_data['accident_count'] > 0, accident_probs > 50)
    death_ml_acc = accuracy_score(processed_data['died'], death_preds)

    # Print accuracies
    print(f"ML Heart Attack Accuracy: {heart_attack_ml_acc:.2%}")
    print(f"ML Accident Accuracy: {accident_ml_acc:.2%}")
    print(f"ML Death Accuracy: {death_ml_acc:.2%}")
    
    # Print confusion matrices
    print("\nConfusion Matrix for Heart Attack Predictions:")
    print(confusion_matrix(processed_data['heart_attack_count'] > 0, heart_attack_probs > 50))

    print("\nConfusion Matrix for Accident Predictions:")
    print(confusion_matrix(processed_data['accident_count'] > 0, accident_probs > 50))

    print("\nConfusion Matrix for Death Predictions:")
    print(confusion_matrix(processed_data['died'], death_preds))

def main():
    file_path = "C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/set2_500_patients.xlsx"
    data = load_data(file_path)
    
    if data is not None:
        processed_data = process_patient_data(data)
        processed_data = engineer_features(processed_data)
        
        # Rule-Based System
        rule_based_df = rule_based_prediction(processed_data.copy())
        
        # Machine Learning System
        heart_attack_features = ['hypertension', 'family_history', 'smoker', 'overweight', 'heart_attack_history', 'heart_attack_risk_score']
        accident_features = ['intoxication_hours', 'low_alertness_hours', 'overweight', 'accident_risk_score']
        
        X_heart_attack = processed_data[heart_attack_features]
        X_accident = processed_data[accident_features]
        y_heart_attack = processed_data['heart_attack_count'] > 0
        y_accident = processed_data['accident_count'] > 0
        
        print("\n--- Training Machine Learning Model for Heart Attack Prediction ---")
        heart_attack_probs, heart_attack_roc_auc, heart_attack_accuracy = train_and_predict(X_heart_attack, y_heart_attack, X_heart_attack)
        heart_attack_probs *= 100
        
        print("\n--- Training Machine Learning Model for Accident Prediction ---")
        accident_probs, accident_roc_auc, accident_accuracy = train_and_predict(X_accident, y_accident, X_accident)
        accident_probs *= 100
        
        # Death prediction combining heart attack and accident risks
        death_probs = (heart_attack_probs * 0.6 + accident_probs * 0.4)
        death_preds = (death_probs > 50).astype(int)
        
        # Print predictions to both terminal and text file
        print_predictions_to_terminal_and_file(processed_data, heart_attack_probs, accident_probs, death_probs, death_preds)
        
        # Print confusion matrices and accuracies
        print_confusion_matrices_and_accuracy(processed_data, heart_attack_probs, accident_probs, death_probs, death_preds)

if __name__ == "__main__":
    main()

