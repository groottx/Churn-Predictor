"""
data_prep.py
Covers FR1 (load/clean), FR2 (EDA helpers), FR3 (feature engineering & split).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BINARY_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
MULTI_CAT_COLS = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'Contract', 'PaymentMethod']


def load_data(path: str) -> pd.DataFrame:
    """FR1: Load the raw CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """FR1: Fix dtypes and handle missing/malformed values."""
    df = df.copy()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    return df


def eda_summary(df: pd.DataFrame) -> dict:
    """FR2: Quick EDA numbers useful for a report (class balance, missingness)."""
    return {
        'shape': df.shape,
        'churn_rate': (df['Churn'] == 'Yes').mean(),
        'null_counts': df.isnull().sum().to_dict(),
        'churn_by_contract': (
            df.groupby('Contract')['Churn']
            .apply(lambda s: (s == 'Yes').mean())
            .to_dict()
        ),
    }


def engineer_features(df: pd.DataFrame):
    """FR3: Encode categoricals, build the model-ready matrix.
    Returns (df_model, feature_names).
    """
    df_model = df.copy()
    df_model['Churn'] = (df_model['Churn'] == 'Yes').astype(int)

    for c in BINARY_COLS:
        df_model[c] = (df_model[c] == 'Yes').astype(int)

    df_model['gender'] = (df_model['gender'] == 'Male').astype(int)

    df_model = pd.get_dummies(df_model, columns=MULTI_CAT_COLS, drop_first=True)

    bool_cols = df_model.select_dtypes(include='bool').columns
    df_model[bool_cols] = df_model[bool_cols].astype(int)

    feature_names = df_model.drop(columns=['Churn']).columns.tolist()
    return df_model, feature_names


def split_and_scale(df_model: pd.DataFrame, feature_names, test_size=0.2, random_state=42):
    """FR3: Stratified train/test split + standardization."""
    X = df_model[feature_names].values.astype(float)
    y = df_model['Churn'].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler


if __name__ == '__main__':
    df = load_data('data/telco.csv')
    df = clean_data(df)
    print(eda_summary(df))

    df_model, feature_names = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_model, feature_names)
    print('Train shape:', X_train.shape, '| Test shape:', X_test.shape)
    print('Num features:', len(feature_names))
