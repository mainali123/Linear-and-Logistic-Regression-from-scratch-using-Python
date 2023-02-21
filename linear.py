import numpy as np
import pandas as pd


def standardize(x):
    return (x - np.mean(x)) / np.std(x)

def cost_function(X, Y, W):
    m = len(Y)
    J = np.sum((X.dot(W) - Y) ** 2)/(2 * m)
    return J

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    for iteration in range(iterations):
        Y_pred = X.dot(B)
        loss = Y_pred - Y
        dw = (X.T.dot(loss) ) / (m)
        B = B - alpha * dw
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
    return B, cost_history


df = pd.read_csv('Fish.csv')
df["Species"] = df["Species"].astype('category')
df["Species"] = df["Species"].cat.codes
df["Weight"] = standardize(df["Weight"])
df["Length1"] = standardize(df["Length1"])
df["Length2"] = standardize(df["Length2"])
df["Length3"] = standardize(df["Length3"])
df["Height"] = standardize(df["Height"])
df["Width"] = standardize(df["Width"])
Species, Weight, Length1, Length2, Length3, Height, Width = df["Species"], df["Weight"], df["Length1"], df["Length2"], df["Length3"], df["Height"], df["Width"]

x0 = np.ones(len(Species))
X2 = np.column_stack([x0,Length1, Length2, Length3, Height, Width])
W = np.array([0, 0,0,0,0,0])
Y2 = np.array(Weight)

inital_cost = cost_function(X2, Y2, W)

alpha = 0.0001
new_weights, cost_history = gradient_descent(X2, Y2, W, alpha, 100000)

Y_pred = X2.dot(new_weights)


print("Enter the vertical length of the fish in cm")
length1 = float(input())
print("Enter the diagonal length of the fish in cm")
length2 = float(input())
print("Enter the cross length of the fish in cm")
length3 = float(input())
print("Enter the height of the fish in cm")
height = float(input())
print("Enter the width of the fish in cm")
width = float(input())

x0 = np.ones(1)
X3 = np.column_stack([x0,length1, length2, length3, height, width])
Y_pred = X3.dot(new_weights)
print("The predicted weight of the fish is", Y_pred)