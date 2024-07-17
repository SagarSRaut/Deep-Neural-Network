import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Load the Titanic dataset
df_titanic = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\MACHINE LEARNING\06 Machine Learning_Sanjay Sane\Repository\Titanic-Dataset.csv")
df_titanic.head()


# Data Preparation
selected_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Survived']
df_titanic = df_titanic[selected_features]
df_titanic = pd.get_dummies(df_titanic, columns=['Sex', 'Embarked'], drop_first=True)

  
imputer = SimpleImputer(strategy='median')
df_titanic['Age'] = imputer.fit_transform(df_titanic[['Age']])

# Split data into features and target variable
X = df_titanic.drop('Survived', axis=1)
y = df_titanic['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier with default parameters
rf_model_default = RandomForestClassifier(random_state=42)
rf_model_default.fit(X_train, y_train)

 
y_pred_default = rf_model_default.predict(X_test)

 
accuracy_default = accuracy_score(y_test, y_pred_default)
print("Accuracy (Default Parameters):", accuracy_default)

# Confusion Matrix
conf_matrix_default = confusion_matrix(y_test, y_pred_default)
print("\nConfusion Matrix (Default Parameters):")
print(conf_matrix_default)

# Tune Parameters using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model_tuned = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model_tuned, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

# predictions with the tuned model
rf_model_tuned = RandomForestClassifier(random_state=42, **best_params)
rf_model_tuned.fit(X_train, y_train)
y_pred_tuned = rf_model_tuned.predict(X_test)

#  Model - Tuned Parameters
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print("\nAccuracy (Tuned Parameters):", accuracy_tuned)

# Confusion Matrix
conf_matrix_tuned = confusion_matrix(y_test, y_pred_tuned)
print("\nConfusion Matrix (Tuned Parameters):")
print(conf_matrix_tuned)

# Save results to a single file
results_df = pd.DataFrame({
    'Metric': ['Accuracy (Default)', 'Accuracy (Tuned)'],
    'Value': [accuracy_default, accuracy_tuned]
})

results_df.to_csv('titanic_results.csv', index=False)

print("\nResults saved to titanic_results.csv")