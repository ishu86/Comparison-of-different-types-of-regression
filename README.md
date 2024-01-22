# Comparison-of-different-types-of-regression
We take a common dataset and apply different kinds of regressions(Linear, Multiple Linear, Polynomial linear, SVR) and observe the results.

Here is the Python code:

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Create the dataset
data = {'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager', 'Country Manager',
                      'Region Manager', 'Partner', 'Senior Partner', 'C-level', 'CEO'],
        'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]}

df = pd.DataFrame(data)

# Prepare the data
X = df[['Level']].values
y = df['Salary'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Multiple Regression
X_multi = df[['Level']].values  # You can add more features here if available
multi_reg = LinearRegression()
multi_reg.fit(X_multi, y)

# Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Support Vector Regression (SVR)
svr_reg = SVR(kernel='linear')
svr_reg.fit(X, y)

# Predictions
y_pred_linear = linear_reg.predict(X_test)
y_pred_multi = multi_reg.predict(X_test)
y_pred_poly = poly_reg.predict(poly_features.transform(X_test))
y_pred_svr = svr_reg.predict(X_test.reshape(-1, 1))

# Evaluation
print("Linear Regression:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_linear))
print("R-squared:", r2_score(y_test, y_pred_linear))
print()

print("Multiple Regression:")
# You can print coefficients and intercept for multiple regression if needed
print()

print("Polynomial Regression:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_poly))
print("R-squared:", r2_score(y_test, y_pred_poly))
print()

print("Support Vector Regression (SVR):")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_svr))
print("R-squared:", r2_score(y_test, y_pred_svr))

# Visualize results for Linear Regression and Polynomial Regression
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, linear_reg.predict(X), color='red', label='Linear Regression')
plt.plot(X, poly_reg.predict(poly_features.transform(X)), color='green', label='Polynomial Regression')
plt.title('Linear vs Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()


# Visualize results for all regressions in a single graph
plt.scatter(X, y, color='blue', label='Actual Data')

# Linear Regression
plt.plot(X, linear_reg.predict(X), color='red', label='Linear Regression')

# Polynomial Regression
plt.plot(X, poly_reg.predict(poly_features.transform(X)), color='green', label='Polynomial Regression')

# SVR Regression
plt.scatter(X_test, y_pred_svr, color='orange', label='SVR Predictions', marker='x')

plt.title('Regression Models Comparison')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()


