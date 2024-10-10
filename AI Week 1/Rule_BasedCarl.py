import pandas as pd
import numpy as np

def load_data(file_path):
    """Load data from the Excel file."""
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        return None

def calculate_heart_attack_risk(patient_data):
    """Calculate heart attack risk score for a patient."""
    score = 0
    if patient_data['hypertension'].any():
        score += 3
    if patient_data['family_history'].any():
        score += 2
    if patient_data['smoker'].any():
        score += 2
    if patient_data['overweight'].any():
        score += 1
    score += 4 * patient_data['heart_attack'].sum()
    return score

def calculate_accident_risk(patient_data):
    """Calculate accident risk score for a patient."""
    score = 0
    score += 3 * (patient_data['intoxication'].sum() / 720) 
    score += 2 * ((patient_data['alertness'] < 0.5).sum() / 720)  
    if patient_data['overweight'].any():
        score += 1
    return score

def calculate_death_risk(heart_attack_risk, accident_risk):
    """Calculate death risk based on heart attack and accident risks."""
    # Assuming death risk is a weighted combination of both risks
    return 0.6 * heart_attack_risk + 0.4 * accident_risk

def process_patient_data(df):
    """Process data for all 500 patients, considering 24 hours per day for 30 days."""
    results = []
    max_heart_attack_score = 0
    max_accident_score = 0

    for i in range(500): 
        patient_data = df.iloc[i*720:(i+1)*720]  
        heart_attack_risk = calculate_heart_attack_risk(patient_data)
        accident_risk = calculate_accident_risk(patient_data)
        
        max_heart_attack_score = max(max_heart_attack_score, heart_attack_risk)
        max_accident_score = max(max_accident_score, accident_risk)
        
        results.append({
            'patient_id': i,
            'heart_attack_risk': heart_attack_risk,
            'accident_risk': accident_risk,
            'hypertension_history': patient_data['hypertension'].any(),
            'family_history': patient_data['family_history'].any(),
            'smoker': patient_data['smoker'].any(),
            'overweight': patient_data['overweight'].any(),
            'heart_attack_count': patient_data['heart_attack'].sum(),
            'accident_count': patient_data['accident'].sum(),
            'intoxication_hours': patient_data['intoxication'].sum(),
            'low_alertness_hours': (patient_data['alertness'] < 0.5).sum(),
            'actual_heart_attack': patient_data['heart_attack'].any(),
            'actual_accident': patient_data['accident'].any(),
            'actual_death': (patient_data['action'] == 'patient died').any()  # Add this line
        })
    
    for result in results:
        result['heart_attack_risk'] = (result['heart_attack_risk'] / max_heart_attack_score) * 100
        result['accident_risk'] = (result['accident_risk'] / max_accident_score) * 100
        result['death_risk'] = calculate_death_risk(result['heart_attack_risk'], result['accident_risk'])
    
    return pd.DataFrame(results)

def calculate_accuracy(processed_data):
    """Calculate accuracy of heart attack, accident, and death predictions."""
    heart_attack_threshold = 50  # Adjust this threshold as needed
    accident_threshold = 50  # Adjust this threshold as needed
    death_threshold = 50  # Adjust this threshold as needed

    heart_attack_correct = sum((processed_data['heart_attack_risk'] > heart_attack_threshold) == processed_data['actual_heart_attack'])
    accident_correct = sum((processed_data['accident_risk'] > accident_threshold) == processed_data['actual_accident'])
    death_correct = sum((processed_data['death_risk'] > death_threshold) == processed_data['actual_death'])

    heart_attack_accuracy = (heart_attack_correct / len(processed_data)) * 100
    accident_accuracy = (accident_correct / len(processed_data)) * 100
    death_accuracy = (death_correct / len(processed_data)) * 100

    return heart_attack_accuracy, accident_accuracy, death_accuracy

def print_patient_predictions(processed_data, output_file):
    """Write predictions for each patient to a file."""
    with open(output_file, 'w') as f:
        for _, patient in processed_data.iterrows():
            f.write(f"Patient {patient['patient_id']}:\n")
            f.write(f"  Heart Attack Risk: {patient['heart_attack_risk']:.2f}%\n")
            f.write(f"  Accident Risk: {patient['accident_risk']:.2f}%\n")
            f.write(f"  Death Risk: {patient['death_risk']:.2f}%\n")
            f.write(f"  Actual Heart Attacks: {patient['heart_attack_count']}\n")
            f.write(f"  Actual Accidents: {patient['accident_count']}\n")
            f.write("  Risk Factors:\n")
            f.write(f"    Hypertension History: {'Yes' if patient['hypertension_history'] else 'No'}\n")
            f.write(f"    Family History: {'Yes' if patient['family_history'] else 'No'}\n")
            f.write(f"    Smoker: {'Yes' if patient['smoker'] else 'No'}\n")
            f.write(f"    Overweight: {'Yes' if patient['overweight'] else 'No'}\n")
            f.write(f"    Hours with Intoxication: {patient['intoxication_hours']}\n")
            f.write(f"    Hours with Low Alertness: {patient['low_alertness_hours']}\n")
            f.write("\n")

def main():
    file_path = "set3_500_patients.xlsx"  # Update this with your file path
    output_file = "patient_predictions.txt"  # New output file
    data = load_data(file_path)
    if data is not None:
        processed_data = process_patient_data(data)
        print_patient_predictions(processed_data, output_file)
        print(f"Patient predictions have been written to {output_file}")
        
        heart_attack_accuracy, accident_accuracy, death_accuracy = calculate_accuracy(processed_data)
        print(f"Heart Attack Prediction Accuracy: {heart_attack_accuracy:.2f}%")
        print(f"Accident Prediction Accuracy: {accident_accuracy:.2f}%")
        print(f"Death Prediction Accuracy: {death_accuracy:.2f}%")

if __name__ == "__main__":
    main()
