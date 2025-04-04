import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import streamlit as st
import warnings

warnings.simplefilter("ignore")

st.title("Titanic Survival Prediction")

# Upload files
train_file = st.file_uploader("Upload Titanic Train CSV", type=["csv"])
test_file = st.file_uploader("Upload Titanic Test CSV", type=["csv"])

if train_file is not None and test_file is not None:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Combine train and test
    data = pd.concat([train_df, test_df], axis=0)

    # Drop unnecessary columns
    data.drop(columns=["Cabin", "Ticket", "Name"], inplace=True)

    # Fix Survived type
    data.iloc[:train_df.shape[0], data.columns.get_loc('Survived')] = data.iloc[:train_df.shape[0], data.columns.get_loc('Survived')].astype(int)

    # Encode categorical
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

    # Impute missing values with median
    for col in ['Age', 'SibSp', 'Parch', 'Fare']:
        data[col].fillna(data[col].median(), inplace=True)

    # Outlier treatment
    data['Age'] = np.where(data['Age'] > 54, 54, np.where(data['Age'] < 3, 3, data['Age']))
    data['SibSp'] = np.where(data['SibSp'] > 2, 2, data['SibSp'])
    data['Parch'] = np.where(data['Parch'] > 0, 0, data['Parch'])
    data['Fare'] = np.where(data['Fare'] > 65, 65, data['Fare'])
    data['Embarked_Q'] = data['Embarked_Q'].astype(int)
    data['Embarked_Q'] = np.where(data['Embarked_Q'] > 0, 0, data['Embarked_Q'])

    # New features
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['Title'] = train_df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
    data['Title'] = data['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')
    data['Title'] = data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev', 'Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Other')
    data = pd.get_dummies(data, columns=['Title'], drop_first=True)

    # Split train/test
    train_data = data.iloc[:train_df.shape[0]].copy()
    test_data = data.iloc[train_df.shape[0]:].copy()

    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']

    # SMOTE for imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Scaling
    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(X_resampled)

    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_resampled, y_resampled)

    st.subheader("Logistic Regression Performance")
    y_pred_log = log_reg.predict(X_resampled)
    st.text(classification_report(y_resampled, y_pred_log))

    # Random Forest with Grid Search
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None, 10],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)
    best_rf = grid_search.best_estimator_

    st.subheader("Random Forest Performance")
    y_pred_rf = best_rf.predict(X_resampled)
    st.text(classification_report(y_resampled, y_pred_rf))

    # Predict on test set
    test_features = test_data.drop(columns=['Survived'])
    test_features_scaled = scaler.transform(test_features)
    test_predictions = best_rf.predict(test_features_scaled)

    st.subheader("Predictions on Test Set")
    st.write(test_predictions)

else:
    st.warning("Please upload both train and test CSV files.")
