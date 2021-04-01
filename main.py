# Import Library
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn

data = pd.read_csv("student-mat.csv", sep=";")

predict = "G3"

data = data[["G1", "G2", "absences","failures", "studytime","G3"]]

# Delete column G3
x = np.array(data.drop([predict], 1))

# Array with only with G3 values
y = np.array(data[predict])

# Distribution of data for training and testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Load mode
linear = linear_model.LinearRegression()
# Model training
linear.fit(x_train, y_train)
# coefficient of determination R^2 of the prediction
acc = linear.score(x_test, y_test)
print(acc)

print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

# predict G3 values of test values
predicted = linear.predict(x_test)

# printing the G3 prediction, X values and the true value of G3
print('predicted:       values:         Original result:')
for x in range(len(predicted)):
    print(predicted[x],  x_test[x], y_test[x])
