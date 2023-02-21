import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes.csv')

# removing outliers from insulin column since it has the most outliers
Q1 = df['Insulin'].quantile(0.25)
Q3 = df['Insulin'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Insulin'] >= Q1 - 1.5 * IQR) & (df['Insulin'] <= Q3 + 1.5 * IQR)]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(yp, y):
    return (-y * np.log(yp) - (1 - y) * np.log(1 - yp)).mean()

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
        z = np.dot(X, B)
        yp = sigmoid(z)
        loss = yp - Y
        dw = np.dot(X.T, loss) / m
        B = B - alpha * dw
        z = np.dot(X, B)
        yp = sigmoid(z)
        cost = cost_function(yp, Y)
        # print("Iteration: ", iteration, " Cost: ", cost)
        cost_history[iteration] = cost
    return B, cost_history

def predict(x, w):
    z = np.dot(x, w)
    y_pred = sigmoid(z)
    return (y_pred > 0.5).astype(int)

x = df.drop('Outcome', axis=1).to_numpy()   # independent variables
y = df['Outcome'].to_numpy()    # dependent variable

x0 = np.ones((x.shape[0], 1))
x = np.concatenate((x0, x), axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12, shuffle=True)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

w = np.zeros(x_train.shape[1])
w, cost_history = gradient_descent(x_train, y_train, w, 0.001, 1000)

y_pred = predict(x_test, w)

def predict_user_input():
    pregnancies = int(input("Enter number of pregnancies: "))
    glucose = int(input("Enter glucose level: "))
    blood_pressure = int(input("Enter blood pressure: "))
    skin_thickness = int(input("Enter skin thickness: "))
    insulin = int(input("Enter insulin level: "))
    bmi = float(input("Enter bmi: "))
    diabetes_pedigree_function = float(input("Enter diabetes pedigree function: "))
    age = int(input("Enter age: "))
    user_input = np.array([1, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
    user_input = user_input.reshape(1, -1)
    y_pred = predict(user_input, w)
    if y_pred == 1:
        print("You have diabetes")
    else:
        print("You don't have diabetes")

predict_user_input()