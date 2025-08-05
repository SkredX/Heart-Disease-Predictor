import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load dataset
df = pd.read_csv("heart_cleveland_upload.csv")

# Preview
display(df.head())

# Data info
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
display(df.describe())

print("Column names in the dataset:")
print(df.columns.tolist())

# Convert 'condition' to binary 'target'
df['target'] = df['condition'].apply(lambda x: 1 if x > 0 else 0)

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='target', y=col, data=df)
    plt.title(f'{col} vs Heart Disease')
    plt.show()

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='target', data=df)
    plt.title(f'{col} vs Heart Disease')
    plt.show()

selected = ['age', 'thalach', 'chol', 'oldpeak', 'target']
sns.pairplot(df[selected], hue='target')
plt.show()

X = df.drop('target', axis=1)
y = df['target']

X = pd.get_dummies(X, columns=['cp', 'slope', 'thal'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Accuracy & Classification Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#ROC-AUC & Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Get feature names back after scaling
feature_names = X.columns
coefficients = model.coef_[0]

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

sns.barplot(data=coef_df, x='Coefficient', y='Feature')
plt.title("Logistic Regression Feature Importance")
plt.show()

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier

# Bagging with Logistic Regression
bag_model = BaggingClassifier(estimator=LogisticRegression(max_iter=1000), n_estimators=100, random_state=42)
bag_model.fit(X_train, y_train)
bag_pred = bag_model.predict(X_test)

# Boosting (Gradient Boosting)
boost_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
boost_model.fit(X_train, y_train)
boost_pred = boost_model.predict(X_test)

# Compare all three models
models = {
    "Logistic Regression": y_pred,
    "Bagging": bag_pred,
    "Boosting": boost_pred
}

for name, preds in models.items():
    print(f"\n{name} Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))
