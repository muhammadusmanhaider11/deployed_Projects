from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train_model import train_logistic
from src.evaluate import evaluate_model

df = load_data("data/loan_data.csv")
X_train, X_test, y_train, y_test = preprocess_data(df)
model = train_logistic(X_train, y_train)
evaluate_model(model, X_test, y_test)