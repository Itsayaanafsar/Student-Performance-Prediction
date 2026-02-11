import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('StudentsPerformance.csv')
x = df[['reading score', 'writing score', 'test preparation course', 'lunch', 'gender', 'race/ethnicity', 'parental level of education']]
y = df['math score']

x_encoded = pd.get_dummies(x, drop_first=True)
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size = 0.2, random_state = 3)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

model2 = RandomForestRegressor(n_estimators=300, random_state=3, max_depth=None, min_samples_leaf=2)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)

sample = x_test.iloc[0:1]
predicted_score = model.predict(sample)[0]

sample2 = x_test.iloc[0:1]
predicted_score2 = model2.predict(sample2)[0]

print("=========Linear Regression model results=========")
print("Mean Absolute Error (Linear Regression): ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (Linear Regression): ", mean_squared_error(y_test, y_pred))
print("R2 Score (Linear Regression): ", r2_score(y_test, y_pred))
print("Actual score: ", y_test.iloc[0])
print("Predicted score: ", predicted_score)
print()

print("=========Random Forest model results=========")
print("Mean Absolute Error (Random Forest): ", mean_absolute_error(y_test, y_pred2))
print("Mean Squared Error (Random Forest): ", mean_squared_error(y_test, y_pred2))
print("R2 Score (Random Forest): ", r2_score(y_test, y_pred2))
print("Actual score: ", y_test.iloc[0])
print("Predicted score: ", predicted_score2)
print()

if r2_score(y_test, y_pred) > r2_score(y_test, y_pred2):
    print("The Linear Regression model performs better than the Random Forest model.")
else:
    print("The Random Forest model performs better than the Linear Regression model.")


# Linear Regression plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.grid(alpha=0.3)
plt.savefig("linear_regression_plot.png", dpi=200)
plt.show()

# Random Forest plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred2, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.grid(alpha=0.3)
plt.savefig("random_forest_plot.png", dpi=200)
plt.show()
