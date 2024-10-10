import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer

def load_data(file_path):
    df = pd.read_excel(file_path)
    # Convert categorical 'gender' column to numeric using one-hot encoding
    df = pd.get_dummies(df, columns=['gender'])
    # Identify numerical columns, excluding gender
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_columns = [col for col in numerical_columns if not col.startswith('gender_')]
    return df, numerical_columns

# Train the model and evaluate using classification report
def train_and_evaluate_model(df, numerical_columns):
    features = numerical_columns + ['gender_female', 'gender_male', 'gender_non-binary']
    target_accident = 'accident'
    target_heart_attack = 'heart_attack'
    
    # Convert data to NumPy arrays for faster processing
    X = df[features].to_numpy()
    y_accident = df[target_accident].to_numpy()
    y_heart_attack = df[target_heart_attack].to_numpy()
    
    # Extract gender columns for fairness analysis
    gender = df[['gender_female', 'gender_male', 'gender_non-binary']].to_numpy()
    
    # Train/test split with NumPy arrays
    X_train, X_val, y_train_accident, y_val_accident, gender_train, gender_val = train_test_split(
        X, y_accident, gender, test_size=0.2, random_state=42
    )
    
    X_train_heart, X_val_heart, y_train_heart, y_val_heart, gender_train_heart, gender_val_heart = train_test_split(
        X, y_heart_attack, gender, test_size=0.2, random_state=42
    )
    
    # Train RandomForestClassifier with NumPy arrays
    rf_accident = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_heart_attack = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rf_accident.fit(X_train, y_train_accident)
    rf_heart_attack.fit(X_train_heart, y_train_heart)
    
    y_val_pred_accident = rf_accident.predict(X_val)
    y_val_pred_heart_attack = rf_heart_attack.predict(X_val_heart)
    
    print("\nAccident Prediction Results:")
    print(classification_report(y_val_accident, y_val_pred_accident))
    
    print("\nHeart Attack Prediction Results:")
    print(classification_report(y_val_heart, y_val_pred_heart_attack))
    
    # Return the correct number of values (9)
    return rf_accident, rf_heart_attack, X_val, y_val_accident, y_val_heart, gender_val, gender_val_heart, features, X_val_heart





# LIME explanation with Matplotlib visualization using NumPy
def lime_explanation(model, X_val, features):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_val, feature_names=features, class_names=['No', 'Yes'], discretize_continuous=True)
    
    # Explain the first prediction as an example
    i = 0
    exp = explainer.explain_instance(X_val[i], model.predict_proba, num_features=len(features))
    
    # Print the LIME explanation as text
    print(exp.as_list())

    # Display the LIME explanation using Matplotlib
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.show()  # Display the plot

# SHAP explanation using a DataFrame for X_val and proper feature names
def shap_explanation(model, X_val, feature_names):
    # Convert X_val back to a DataFrame with feature names
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val_df)
    
    # SHAP summary plot using DataFrame for better feature representation
    shap.summary_plot(shap_values[1], X_val_df, feature_names=feature_names)
# Fairness analysis with Fairlearn using NumPy arrays for X_val and gender
def fairness_analysis(model, X_val, y_val, gender):
    # Ensure gender has the same shape as X_val
    if len(gender) != len(X_val):
        raise ValueError(f"Inconsistent number of samples: X_val has {len(X_val)} rows, but gender has {len(gender)} rows.")

    # Calculate fairness metrics
    metric_frame = MetricFrame(metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
                               y_true=y_val, y_pred=model.predict(X_val), sensitive_features=gender)
    
    print(metric_frame.by_group)
    
    # Compute demographic parity and equalized odds
    disparity = demographic_parity_difference(y_true=y_val, y_pred=model.predict(X_val), sensitive_features=gender)
    equalized_odds = equalized_odds_difference(y_true=y_val, y_pred=model.predict(X_val), sensitive_features=gender)
    
    print(f"Demographic Parity Difference: {disparity}")
    print(f"Equalized Odds Difference: {equalized_odds}")


# Main execution
file_path = 'C:/Users/alenq/Documents/Computer_Science_Course_UM/repository year 2/Intro-To-AI/AI Week 6/Folder/set4_500_patients.xlsx'
df, numerical_columns = load_data(file_path)

# Train the model
rf_accident, rf_heart_attack, X_val, y_val_accident, y_val_heart, gender, features = train_and_evaluate_model(df, numerical_columns)

# Explainability - LIME for Accident prediction with Matplotlib visualization
lime_explanation(rf_accident, X_val, features)
# Explainability - SHAP for Heart Attack prediction
shap_explanation(rf_heart_attack, X_val, features)

# Unpack the correct number of values (9 not 7)
rf_accident, rf_heart_attack, X_val, y_val_accident, y_val_heart, gender_val, gender_val_heart, features, X_val_heart = train_and_evaluate_model(df, numerical_columns)


# Fairness analysis for Accidents
print("Fairness analysis for Accidents:")
fairness_analysis(rf_accident, X_val, y_val_accident, gender_val)

# Fairness analysis for Heart Attacks
print("Fairness analysis for Heart Attacks:")
fairness_analysis(rf_heart_attack, X_val_heart, y_val_heart, gender_val_heart)
 
