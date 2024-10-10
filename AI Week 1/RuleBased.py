def predict_risk(patient):
    heart_attack_risk = 0
    accident_risk = 0
    death_risk = 0
    
    # Heart attack risk rules
    if patient['hypertension_score'] > 7:
        heart_attack_risk += 1
    if patient['history_heart_attack']:
        heart_attack_risk += 1
    if patient['family_history'] or patient['smokes'] or patient['overweight']:
        heart_attack_risk += 1

    # Accident risk rules
    if patient['intoxication'] > 8:
        accident_risk += 1
    if patient['alertness'] < 3 or patient['activity'] == 'drink alcohol':
        accident_risk += 1
    if patient['overweight'] or patient['history_accident']:
        accident_risk += 1
    
    # Death prediction
    if heart_attack_risk >= 3:
        death_risk += 1
    if accident_risk >= 2:
        death_risk += 1
    
    return heart_attack_risk, accident_risk, death_risk
