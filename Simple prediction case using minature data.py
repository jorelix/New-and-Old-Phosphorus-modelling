import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Example data (replace with your real arrays)
Q = np.array([10, 20, 30, 40, 50], dtype=float)
t = np.linspace(0, 1, len(Q))  # example time values
L = np.array([5, 15, 25, 35, 45], dtype=float)

# 1. Build feature matrix
X = np.column_stack([
    np.log(Q),              # log(Q)
    np.sin(2 * np.pi * t),  # sin(2πt)
    np.cos(2 * np.pi * t)   # cos(2πt)
])

# 2. Response variable (log(L))
y = np.log(L)

# 3. Fit linear regression
model = LinearRegression()
model.fit(X, y)

# 4. Extract coefficients
beta1, beta2, beta3 = model.coef_
beta0 = model.intercept_

print("β0 =", beta0)
print("β1 =", beta1)
print("β2 =", beta2)
print("β3 =", beta3)

# 5. Predictions
y_pred = model.predict(X)