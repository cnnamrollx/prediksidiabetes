# Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Load dataset
df = pd.read_csv('diabetes.csv')

# Tampilkan informasi dasar dataset
print("Shape of dataset:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nStatistik Deskriptif:")
print(df.describe())

# Cek nilai yang hilang
print("\nNilai yang hilang:")
print(df.isnull().sum())

# Cek distribusi target
print("\nDistribusi Target (Outcome):")
print(df['Outcome'].value_counts())
print("Persentase:")
print(df['Outcome'].value_counts(normalize=True) * 100)

# Pisahkan fitur dan target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data menjadi training dan testing set (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisasi fitur (opsional namun direkomendasikan)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Buat model Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,        # Jumlah pohon
    max_depth=10,            # Kedalaman maksimum pohon
    min_samples_split=5,     # Minimum sampel untuk split
    min_samples_leaf=2,      # Minimum sampel di leaf node
    random_state=42,
    n_jobs=-1               # Gunakan semua CPU cores
)

# Train model
print("\nTraining Random Forest model...")
rf_model.fit(X_train_scaled, y_train)

# Prediksi
y_pred = rf_model.predict(X_test_scaled)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print("\nFeature Importance:")
print(feature_importance)

# Hyperparameter Tuning (Opsional)
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search dengan cross-validation
print("\nMelakukan Hyperparameter Tuning...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluasi model terbaik
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Test accuracy with best model: {accuracy_best:.4f}")

# Save metrics
metrics = {
    'accuracy': accuracy_best,
    'classification_report': classification_report(y_test, y_pred_best, output_dict=True),
    'confusion_matrix': cm.tolist(),  # from earlier cm
    'cv_score': grid_search.best_score_
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)
print("\nMetrics saved to metrics.json")
# Save the scaler
joblib.dump(scaler, 'scaler.joblib')
print("\nScaler saved to scaler.joblib")
# Save the best model
joblib.dump(best_model, 'rf_diabetes_model.joblib')
print("\nBest model saved to rf_diabetes_model.joblib")
# Predict on entire dataset using best model
X_scaled = scaler.transform(X)
df['Predicted_Outcome'] = best_model.predict(X_scaled)
df.to_csv('diabetes_predictions.csv', index=False)
print("\nPredictions saved to diabetes_predictions.csv")