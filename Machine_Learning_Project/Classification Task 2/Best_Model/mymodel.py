from email.charset import add_alias

import joblib
import numpy as np
import pandas as pd
from extra_functions import Normalize, CubicSplines, InterpolData, SaySide, KeepPoints, ActiveSide

model_E12 = joblib.load("model_E12.pkl")
model_E3 = joblib.load("model_E3.pkl")
model_E4 = joblib.load("model_E4.pkl")
model_E5 = joblib.load("model_E5.pkl")


def predict(data_test):
    data_test_normalized = data_test.copy()
    data_test_normalized['Skeleton_Sequence'] = data_test['Skeleton_Sequence'].apply(lambda seq: Normalize(np.array(seq)))

    X_interp_5 = InterpolData(data_test_normalized, 5)
    exercise_types = X_interp_5['Exercise_Id'].unique()
    exercise_dfs = {ex: X_interp_5[X_interp_5['Exercise_Id'] == ex] for ex in exercise_types}

    # E1 and E2
    E12_pos = pd.concat([exercise_dfs['E1'], exercise_dfs['E2']], ignore_index=True)

    PosListE12 = []
    for i, seq in E12_pos.iterrows():
        feats = KeepPoints(seq['Skeleton_Sequence'], ActiveSide(seq['Skeleton_Sequence']))
        PosListE12.append(feats)

    X_E12_pos = np.array(PosListE12)
    X_E12_pos_flat = X_E12_pos.reshape(X_E12_pos.shape[0], -1)

    patients_E12_pos = E12_pos['Patient_Id'].values

    # Exercise 3
    E3_pos = exercise_dfs['E3']

    joints_E3 = [11, 12, 13, 14, 15, 16]

    PosListE3 = []
    for i, seq in E3_pos.iterrows():
        feats = KeepPoints(seq['Skeleton_Sequence'], joints_E3)
        PosListE3.append(feats)

    X_E3_pos = np.array(PosListE3)
    X_E3_pos_flat = X_E3_pos.reshape(X_E3_pos.shape[0], -1)
    X_E3_pos_flat = np.hstack((X_E3_pos_flat, SaySide(E3_pos).reshape(-1, 1)))

    patients_E3_pos = E3_pos['Patient_Id'].values

    # Exercise 4
    E4_pos = exercise_dfs['E4']

    joints_E4 = [13, 14, 15, 16]

    PosListE4 = []
    for i, seq in E4_pos.iterrows():
        feats = KeepPoints(seq['Skeleton_Sequence'], joints_E4)
        PosListE4.append(feats)

    X_E4_pos = np.array(PosListE4)
    X_E4_pos_flat = X_E4_pos.reshape(X_E4_pos.shape[0], -1)

    patients_E4_pos = E4_pos['Patient_Id'].values

    # Exercise 5
    E5_pos = exercise_dfs['E5']

    joints_E5 = [11, 12, 15, 16, 25, 26]

    PosListE5 = []
    for i, seq in E5_pos.iterrows():
        feats = KeepPoints(seq['Skeleton_Sequence'], joints_E5)
        PosListE5.append(feats)

    X_E5_pos = np.array(PosListE5)
    X_E5_pos_flat = X_E5_pos.reshape(X_E5_pos.shape[0], -1)
    X_E5_pos_flat = np.hstack((X_E5_pos_flat, SaySide(E5_pos).reshape(-1, 1)))

    patients_E5_pos = E5_pos['Patient_Id'].values

    unique_patients = np.unique(patients_E12_pos)

    y_pred_E12_prob = model_E12.predict_proba(X_E12_pos_flat)[:, 1]
    y_pred_E12_patient = []
    y_pred_E12_patient_prob = []

    y_pred_E3_prob = model_E3.predict_proba(X_E3_pos_flat)[:, 1]
    y_pred_E3_patient = []
    y_pred_E3_patient_prob = []

    y_pred_E4_prob = model_E4.predict_proba(X_E4_pos_flat)[:, 1]
    y_pred_E4_patient = []
    y_pred_E4_patient_prob = []

    y_pred_E5_prob = model_E5.predict_proba(X_E5_pos_flat)[:, 1]
    y_pred_E5_patient = []
    y_pred_E5_patient_prob = []

    for patient in unique_patients:
        maskE12 = patients_E12_pos == patient
        mean_probE12 = np.mean(y_pred_E12_prob[maskE12])
        y_pred_E12_patient_prob.append(mean_probE12)

        maskE3 = patients_E3_pos == patient
        mean_probE3 = np.mean(y_pred_E3_prob[maskE3])
        y_pred_E3_patient_prob.append(mean_probE3)

        maskE4 = patients_E4_pos == patient
        mean_probE4 = np.mean(y_pred_E4_prob[maskE4])
        y_pred_E4_patient_prob.append(mean_probE4)

        maskE5 = patients_E5_pos == patient
        mean_probE5 = np.mean(y_pred_E5_prob[maskE5])
        y_pred_E5_patient_prob.append(mean_probE5)

    probs_stack = np.stack([np.array(y_pred_E12_patient_prob), np.array(y_pred_E3_patient_prob),
                            np.array(y_pred_E4_patient_prob), np.array(y_pred_E5_patient_prob)], axis=1)

    weights = np.array([0.3533992459382818, 0.07885403034254317, 0.21434747778089328, 0.3533992459382818])

    weighted_probs = np.dot(probs_stack, weights)

    y_pred = (weighted_probs >= 0.5).astype(int)

    return y_pred