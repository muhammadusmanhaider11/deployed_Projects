# Loan Default Prediction Using Logistic Regression

This project implements a machine learning model to predict loan default risk using logistic regression. It includes data preprocessing, model training, evaluation, and a web interface for making predictions.

## Project Structure

```
loan_default_prediction/
├── app/
│   └── streamlit_app.py      # Web interface for predictions
├── data/
│   ├── loan_data.csv         # Original dataset
│   └── visualizations/       # Data visualization plots
├── models/
│   └── logistic_model.pkl    # Trained model
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis
├── src/
│   ├── data_loader.py       # Data loading utilities
│   ├── evaluate.py          # Model evaluation
│   ├── preprocess.py        # Data preprocessing
│   ├── train_model.py       # Model training
│   ├── utils.py            # Utility functions
│   └── visualize.py        # Visualization functions
├── main.py                  # Main script to run the pipeline
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Features Used

The model uses the following features to predict loan default risk:

### Numerical Features:
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term

### Categorical Features:
- Gender
- Marital Status
- Number of Dependents
- Education Level
- Self Employment Status
- Credit History
- Property Area

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd loan_default_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

Run the main script to load data, preprocess it, train the model, and generate visualizations:

```bash
python main.py
```

This will:
- Load the loan dataset
- Create visualizations of the data distribution
- Preprocess the data (handle missing values, encode categorical variables)
- Train a logistic regression model
- Evaluate the model's performance
- Save the trained model

### 2. Using the Web Interface

Start the Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

The web interface allows you to:
- Input loan application details
- Get instant predictions about default risk
- View the probability of default

## Model Performance

The current model achieves the following performance metrics:

- **Accuracy**: 79%
- **ROC-AUC**: 0.75
- **Precision** (No Default): 95%
- **Recall** (Default): 99%

### Detailed Metrics:
```
              precision    recall  f1-score   support
           0       0.95      0.42      0.58        43
           1       0.76      0.99      0.86        80
```

## Data Visualizations

The project generates several visualizations to help understand the data:

1. **Before Preprocessing**:
   - Numerical feature distributions (`before_numerical_distributions.png`)
   - Categorical feature distributions (`before_categorical_distributions.png`)
   - Feature correlation matrix (`before_correlation_matrix.png`)

2. **After Preprocessing**:
   - Processed feature correlation matrix (`after_preprocessing_correlation.png`)

These visualizations can be found in the `data/visualizations/` directory.

## Preprocessing Steps

1. **Handling Missing Values**:
   - Numerical features: Filled with mean values
   - Categorical features: Filled with mode values

2. **Feature Engineering**:
   - Categorical variables are one-hot encoded
   - Numerical features are standardized using StandardScaler

## Model Description

The project uses Logistic Regression for prediction because:
- It's well-suited for binary classification problems
- Provides interpretable results
- Performs well with limited data
- Offers probability scores for predictions

## Web Interface Features

The Streamlit interface provides:
1. Input fields for all loan application details
2. Real-time predictions
3. Probability scores for default risk
4. User-friendly interface

## Dependencies

Main libraries used:
- pandas
- scikit-learn
- streamlit
- matplotlib
- seaborn

See `requirements.txt` for complete list.

## Future Improvements

Potential areas for enhancement:
1. Feature engineering to create more predictive variables
2. Experimenting with other algorithms (Random Forest, XGBoost)
3. Adding more detailed visualizations in the web interface
4. Implementing cross-validation
5. Adding model explanation using SHAP or LIME

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
