# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and separate input feature (Level) and target variable (Salary).

2.Split the dataset into training and testing sets.

3.Train a Decision Tree Regressor model using the training data.

4.Predict salary values and evaluate model performance using error metrics.
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S AVAN ARUL
RegisterNumber: 212225040036
*/
```
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv(r"C:\Users\acer\Downloads\Salary.csv")

X = df[["Level"]].values
y = df["Salary"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)


plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=["Level"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Regressor for Employee Salary Prediction")
plt.show()


new_exp = [[5]]
predicted_salary = model.predict(new_exp)

print("Predicted Salary for 5 years experience:", predicted_salary[0])
~~~

## Output:
<img width="1246" height="100" alt="image" src="https://github.com/user-attachments/assets/cc515ea9-e12e-47c0-92f2-cc742f33cdc3" />
<img width="1240" height="690" alt="image" src="https://github.com/user-attachments/assets/e491a0ed-9cfc-4673-8bdc-9b47f2e48c10" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
