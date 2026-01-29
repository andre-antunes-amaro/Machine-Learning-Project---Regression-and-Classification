import joblib
import numpy as np
from extra_functions import Normalize, Flip, SaySide

model = joblib.load("final_model.pkl")

def predict(X_test):
    X_test_processed = np.hstack([Flip(Normalize(X_test.copy())), SaySide(X_test)])
    return model.predict(X_test_processed)