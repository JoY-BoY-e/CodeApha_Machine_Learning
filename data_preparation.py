import streamlit as st
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global storage for training features
TRAIN_FEATURES = []
scaler = StandardScaler()

def clean_lendingclub_data(df, is_train=True):
    global TRAIN_FEATURES, scaler

    # Step 1: Clean Column Names
    df.columns = df.columns.str.strip()

    # Step 2: Remove Irrelevant Columns
    irrelevant_cols = ['zip_code', 'id', 'member_id', 'url', 'desc', 'title', 'policy_code']
    df = df.drop(columns=[col for col in irrelevant_cols if col in df.columns], errors='ignore')

    # Step 3: Handle Missing Values
    threshold = 0.5
    df = df.loc[:, df.isnull().mean() < threshold]

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Step 4: Handle Target Variable
    if is_train:
        if 'loan_status' in df.columns:
            if df['loan_status'].dtype not in ['int64', 'float64']:
                df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]
                df['loan_status'] = df['loan_status'].replace({
                    'Fully Paid': 1,
                    'Charged Off': 0,
                    'Default': 0
                })
        else:
            st.error("'loan_status' column not found. Cannot proceed.")
            return None
    else:
        if 'loan_status' in df.columns:
            df = df.drop(columns=['loan_status'])  # remove during prediction

    # Step 5: Feature Engineering
    def clean_emp_length(x):
        if x == '< 1 year':
            return 0
        elif x == '10+ years':
            return 10
        try:
            return int(str(x).strip().split()[0])
        except:
            return np.nan

    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].apply(clean_emp_length)
        df['emp_length'] = df['emp_length'].fillna(df['emp_length'].median())

    # Handle 'term' feature
    if 'term' in df.columns:
        df['term'] = df['term'].str.extract('(\d+)').astype(float)
    else:
        # Try building it from one-hot encoding if applicable
        term_cols = [col for col in df.columns if 'term' in col]
        if 'term_60 months' in term_cols and 'term_36 months' in term_cols:
            df['term'] = df['term_60 months'] * 60 + df['term_36 months'] * 36
        elif 'term_ 60 months' in term_cols and 'term_ 36 months' in term_cols:
            df['term'] = df['term_ 60 months'] * 60 + df['term_ 36 months'] * 36
        else:
            df['term'] = 36  # default to 36 months if not found

    # Step 6: Encode Categorical Variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'loan_status' in categorical_cols:
        categorical_cols.remove('loan_status')

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align with training features
    if is_train:
        TRAIN_FEATURES = df.columns.tolist()
    else:
        # Add missing cols
        for col in TRAIN_FEATURES:
            if col not in df.columns:
                df[col] = 0
        # Drop extra cols
        df = df[TRAIN_FEATURES]

    # Step 7: Feature Scaling
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if is_train:
        if 'loan_status' in numerical_cols:
            numerical_cols.remove('loan_status')
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        if 'loan_status' in numerical_cols:
            numerical_cols.remove('loan_status')
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df


# Example training function
def train_model(train_df):
    df = clean_lendingclub_data(train_df, is_train=True)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    model = LogisticRegression()
    model.fit(X, y)

    # Save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump((model, TRAIN_FEATURES, scaler), f)

    return model


# Example prediction function
def predict_model(new_data):
    with open('model.pkl', 'rb') as f:
        model, TRAIN_FEATURES_SAVED, scaler_saved = pickle.load(f)

    global TRAIN_FEATURES, scaler
    TRAIN_FEATURES = TRAIN_FEATURES_SAVED
    scaler = scaler_saved

    df = clean_lendingclub_data(new_data, is_train=False)
    predictions = model.predict(df)
    return predictions
