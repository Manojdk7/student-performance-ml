import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load dataset (tab-separated)
data = pd.read_csv("studentsperformace.csv", sep="\t")

# Clean column names
data.columns = data.columns.str.strip()

print("Columns:", data.columns)

# Select multiple features (IMPORTANT improvement)
X = data[['Study_Hours_per_Week',
          'Attendance',
          'Assignments_Avg',
          'Quizzes_Avg',
          'Projects_Score']]

y = data['Final_Score']

# Train-test split (professional way)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predicted_scores = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predicted_scores)
r2 = r2_score(y_test, predicted_scores)
mae = mean_absolute_error(y_test, predicted_scores)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {round(mae, 2)}')
print(f'Mean Squared Error: {round(mse, 2)}')
print(f'R-squared: {round(r2, 2)}')
print(f'Root Mean Squared Error: {round(rmse, 2)}')

# -------------------------------
# Histogram
# -------------------------------
plt.figure(figsize=(10, 6))
plt.hist(data['Final_Score'], bins=20)
plt.title('Distribution of Final Scores')
plt.xlabel('Final Score')
plt.ylabel('Number of Students')
plt.grid(True)
plt.show()

# -------------------------------
# Scatter + Regression (only for Study Hours)
# -------------------------------
plt.figure(figsize=(10, 6))

# Train simple model for visualization (1 feature only)
simple_model = LinearRegression()
simple_model.fit(data[['Study_Hours_per_Week']], data['Final_Score'])

simple_pred = simple_model.predict(data[['Study_Hours_per_Week']])

# Sort for smooth line
sorted_index = data['Study_Hours_per_Week'].argsort()

plt.scatter(data['Study_Hours_per_Week'], data['Final_Score'], label='Actual')

plt.plot(data['Study_Hours_per_Week'].iloc[sorted_index],
         simple_pred[sorted_index],
         color='black',
         linewidth=2,
         label='Regression Line')

plt.title('Study Hours vs Final Score')
plt.xlabel('Study Hours per Week')
plt.ylabel('Final Score')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Prediction for new student
# -------------------------------
new_data = pd.DataFrame([[9, 85, 75, 70, 80]],
columns=['Study_Hours_per_Week',
         'Attendance',
         'Assignments_Avg',
         'Quizzes_Avg',
         'Projects_Score'])

predicted_new_score = model.predict(new_data)

print(f'Predicted score: {round(predicted_new_score[0], 2)}')