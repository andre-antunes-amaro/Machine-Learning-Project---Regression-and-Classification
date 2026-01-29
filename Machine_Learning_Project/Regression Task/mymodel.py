import joblib

model = joblib.load("final_model.pkl")

def predict(X_test):
    return model.predict(X_test)