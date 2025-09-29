from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def preprocess_data(df, return_preprocessed=False):
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Drop Loan_ID as it's not needed for prediction
    df_processed = df_processed.drop('Loan_ID', axis=1)
    
    # Handle missing values
    numeric_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    
    # Fill numeric missing values with mean
    for col in numeric_columns:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    # Fill categorical missing values with mode
    for col in categorical_columns:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    # Store the preprocessed data before dummy variables for visualization
    df_before_dummies = df_processed.copy()
    
    # Create dummy variables for categorical columns
    df_processed = pd.get_dummies(df_processed, columns=categorical_columns, drop_first=True)
    
    # Separate features and target
    X = df_processed.drop('Loan_Status', axis=1)
    y = df_processed['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    if return_preprocessed:
        # Return both the split data and the preprocessed dataframe
        preprocessed_df = pd.DataFrame(X_scaled, columns=X.columns)
        preprocessed_df['Loan_Status'] = y
        return X_train, X_test, y_train, y_test, preprocessed_df
    
    return X_train, X_test, y_train, y_test