import numpy as np

# Normalizes data according to torso and shoulder length
def Normalize(X):
    X_raw = X.copy()

    Avg = X_raw[:, :66]
    Std = X_raw[:, 66:]
    n = Avg.shape[0]

    Avg = Avg.reshape(n, 33, 2)
    Std = Std.reshape(n, 33, 2)

    # correct the sign of the y coordinates
    Avg[:, :, 1] *= -1

    sw = np.linalg.norm(Avg[:, 11, :] - Avg[:, 12, :], axis=1) # shoulder width
    th = np.linalg.norm(Avg[:, 0, :] - 0.5 * (Avg[:, 23, :] + Avg[:, 24, :]), axis=1) # torso height

    scale = sw / th

    Avg = Avg / scale[:, None, None]
    Std = Std / scale[:, None, None]

    Avg = Avg.reshape(n, -1)
    Std = Std.reshape(n, -1)
    return np.hstack([Avg, Std])

# Ensures all movements are represented as if performed with the positive side of the body
def Flip(X):
    X_raw = X.copy()

    Avg = X_raw[:, :66]
    Std = X_raw[:, 66:]
    n = Avg.shape[0]

    Avg = Avg.reshape(n, 33, 2)
    Std = Std.reshape(n, 33, 2)

    L = 0
    for i in range(n):
      # Condition for swapping indexes
      if np.linalg.norm(Std[i,12::2]) > np.linalg.norm(Std[i,11::2]):
        AvgSwapped = Avg[i].copy()
        AvgSwapped[2::2], AvgSwapped[1::2] = Avg[i, 1::2], Avg[i, 2::2]
        Avg[i] = AvgSwapped
        Avg[i, :, 0] *= -1

        StdSwapped = Std[i].copy()
        StdSwapped[2::2], StdSwapped[1::2] = Std[i, 1::2], Std[i, 2::2]
        Std[i] = StdSwapped
        L += 1
    Avg = Avg.reshape(n, -1)
    Std = Std.reshape(n, -1)
    return np.hstack([Avg, Std])

# Informs which side the movement was executed at
def SaySide(X):
    X_raw = X.copy()

    Avg = X_raw[:, :66]
    Std = X_raw[:, 66:]
    n = Avg.shape[0]

    Avg = Avg.reshape(n, 33, 2)
    Std = Std.reshape(n, 33, 2)

    # -1 for right-dominant, +1 for left-dominant
    Side = np.array([
        -1 if np.linalg.norm(Std[i, 12::2]) > np.linalg.norm(Std[i, 11::2]) else 1
        for i in range(n)
    ])

    return Side[:, None]