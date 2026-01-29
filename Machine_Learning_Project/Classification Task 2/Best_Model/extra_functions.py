import numpy as np
from scipy.interpolate import splrep, splev

def Normalize(seq):
  n = seq.shape[0]
  X = seq.reshape(n, 33, 2).copy()

  X[:, :, 1] *= -1 # correct the sign of the y coordinates

  sw = np.linalg.norm(X[:, 11, :] - X[:, 12, :], axis=1) # shoulder width
  th = np.linalg.norm(X[:, 0, :] - 0.5 * (X[:, 23, :] + X[:, 24, :]), axis=1) # torso height

  scale = sw / th
  X = X / scale[:, None, None]
  X = X.reshape(n, -1)
  return X

def CubicSplines(seq, n):
  # n is the lenght of the new sequences
    T = seq.shape[0]
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, n)

    interp = np.zeros((n, seq.shape[1], seq.shape[2]))

    for j in range(seq.shape[1]):  # joints
        for k in range(seq.shape[2]):  # x and y
            tck = splrep(old_t, seq[:, j, k], s = 0)
            interp[:, j, k] = splev(new_t, tck)

    return interp

# Applies interpolation to the whole data
def InterpolData(data, n):
    data = data.copy()
    data['Skeleton_Sequence'] = data['Skeleton_Sequence'].apply(lambda x: CubicSplines(np.array(x).reshape(-1, 33, 2), n))
    return data

def SaySide(df):
  n = df.shape[0]
  Sides = np.ones(n)

  for i in range(n):
      seq = np.array(df.iloc[i]['Skeleton_Sequence'])
      m = seq.shape[0]
      seq = seq.reshape(m, 33, 2)

      Std = seq.std(axis=0)
      if np.linalg.norm(Std[12::2, :]) > np.linalg.norm(Std[11::2, :]):
        Sides[i] = -1

  return Sides

def ActiveSide(seq):
  std_left_wrist = seq[:, 15, 0].std(axis=0) + seq[:, 15, 1].std(axis=0)
  std_right_wrist = seq[:, 16, 0].std(axis=0) + seq[:, 16, 1].std(axis=0)
  std_left_elbow = seq[:, 13, 0].std(axis=0) + seq[:, 13, 1].std(axis=0)
  std_right_elbow = seq[:, 14, 0].std(axis=0) + seq[:, 14, 1].std(axis=0)

  if std_left_wrist + std_left_elbow > std_right_wrist + std_right_elbow:
    return [11, 12, 15, 13]
  else:
    return [11, 12, 16, 14]

def KeepPoints(seq, joints):
  return seq[:, joints, :]