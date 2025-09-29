import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_dataset_distributions(df, title_prefix="Before"):
    """Plot distributions of numerical features and count plots of categorical features"""
    
    # Set the style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Numerical features
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    # Create distribution plots for numerical features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df, x=col, hue='Loan_Status', bins=30)
        plt.title(f'{title_prefix} Preprocessing: {col} Distribution')
    plt.tight_layout()
    plt.savefig(f'data/visualizations/{title_prefix.lower()}_numerical_distributions.png')
    plt.close()
    
    # Categorical features
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                       'Self_Employed', 'Credit_History', 'Property_Area']
    
    # Create count plots for categorical features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(3, 3, i)
        sns.countplot(data=df, x=col, hue='Loan_Status')
        plt.xticks(rotation=45)
        plt.title(f'{title_prefix} Preprocessing: {col} Distribution')
    plt.tight_layout()
    plt.savefig(f'data/visualizations/{title_prefix.lower()}_categorical_distributions.png')
    plt.close()
    
    # Correlation matrix for numerical features
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_cols + ['Credit_History']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'{title_prefix} Preprocessing: Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'data/visualizations/{title_prefix.lower()}_correlation_matrix.png')
    plt.close()

def plot_preprocessing_comparison(df_before, df_after):
    """Plot comparison of data before and after preprocessing"""
    
    # Create visualizations directory if it doesn't exist
    import os
    os.makedirs('data/visualizations', exist_ok=True)
    
    # Plot distributions before preprocessing
    plot_dataset_distributions(df_before, "Before")
    
    # For after preprocessing, we'll plot the correlation matrix of the processed features
    plt.figure(figsize=(12, 10))
    correlation_matrix = pd.DataFrame(df_after).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('After Preprocessing: Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('data/visualizations/after_preprocessing_correlation.png')
    plt.close()