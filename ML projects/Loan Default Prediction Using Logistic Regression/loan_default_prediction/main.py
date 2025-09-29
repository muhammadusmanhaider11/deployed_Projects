from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train_model import train_logistic
from src.evaluate import evaluate_model
from src.visualize import plot_dataset_distributions, plot_preprocessing_comparison
import numpy as pd

def main():
    print("Loading data...")
    df = load_data('data/loan_data.csv')
    
    print("Creating visualizations before preprocessing...")
    plot_dataset_distributions(df, "Before")
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessed_data = preprocess_data(df, return_preprocessed=True)
    
    print("Creating visualizations after preprocessing...")
    plot_preprocessing_comparison(df, preprocessed_data)
    
    print("Training model...")
    model = train_logistic(X_train, y_train)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()