# churn_pipeline.py

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("customer_churn.csv")
print("✅ Data loaded. Rows:", len(df))

# Drop customerID column
df.drop('customerID', axis=1, inplace=True)

# Replace blanks with NaN and drop missing
df.replace(" ", np.nan, inplace=True)
df.dropna(inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Print quick preview
print("\n🔍 Preview of cleaned data:")
print(df.head())

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Encode categorical features
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

print("\n✅ Data encoded and split.")
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained. Accuracy: {acc:.4f}")

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

import joblib

# Save the trained model to disk
joblib.dump(model, "churn_model.pkl")
print("\n💾 Model saved as churn_model.pkl")

# Create and save a CSV report of predictions
output = pd.DataFrame(X_test, columns=X.columns)
output["Actual"] = y_test.values
output["Predicted"] = y_pred
output.to_csv("report_output.csv", index=False)
print("📝 Prediction report saved as report_output.csv")
