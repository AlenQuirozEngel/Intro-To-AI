from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Assuming 'data' is a DataFrame with your patient features and 'labels' contains the target (heart attack, accident, death)
X = data[['hypertension_score', 'alertness', 'intoxication', 'activity', 'history_heart_attack', 'history_accident']]
y = labels

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
