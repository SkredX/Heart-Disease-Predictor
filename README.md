# Heart Disease Prediction using Logistic Regression

This project builds a predictive model to detect heart disease using patient data and Logistic Regression. It involves exploratory data analysis (EDA), preprocessing, model training, and evaluation using various metrics and ensemble techniques.

## Dataset
- **Source**: [Heart Disease UCI - Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- **Features**:
  - age, sex, chest pain type (cp), resting blood pressure (trestbps), serum cholesterol (chol), fasting blood sugar (fbs), resting electrocardiographic results (restecg), max heart rate (thalach), exercise-induced angina (exang), ST depression (oldpeak), slope, number of major vessels (ca), thalassemia (thal)
  - **Target**: `condition` (0 = no disease, 1 = disease)

## Workflow
1. **Data Loading and Cleaning**  
   Handle missing values, convert data types, and drop irrelevant entries.

2. **Exploratory Data Analysis (EDA)**  
   Visualize data distributions, correlations, and relationships among features.

3. **Preprocessing**  
   One-hot encoding, feature scaling, and train-test split.

4. **Model Training**  
   Train a Logistic Regression model with hyperparameter tuning.

5. **Evaluation**  
   Use confusion matrix, accuracy, ROC-AUC curve.

6. **Ensemble Models**  
   Bagging and Gradient Boosting to enhance model robustness and compare results.

## Key Tools Used
- Python (Jupyter Notebook)
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (LogisticRegression, BaggingClassifier, GradientBoostingClassifier, etc.)

## Result
Achieved a high-performing Logistic Regression model with additional ensemble comparisons for improved reliability.
