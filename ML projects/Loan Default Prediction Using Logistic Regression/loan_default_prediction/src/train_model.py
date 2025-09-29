from sklearn.linear_model import LogisticRegression
import joblib

def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/logistic_model.pkl')
    return model