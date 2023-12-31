import numpy as np
import pandas as pd
import utils


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y , lambda_param):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    grad = (1 / m) * np.dot(X.T, (h - y))

    grad[1:] += (lambda_param / m) * theta[1:]

    return J, grad


def gradient_descent(theta, X, y, alpha, lambda_param, num_iters):
    J_history = []
    m = len(y)

    for i in range(num_iters):
        cost, grad = cost_function(theta, X, y, lambda_param)
        theta -= alpha * grad
        J_history.append(cost)
        
    return theta, J_history


def train(X, y, alpha, lambda_param, num_iters):
    m, n = X.shape
    theta = np.zeros(n)

    theta, J_history = gradient_descent(theta, X, y, alpha, lambda_param, num_iters)
        
    return theta, J_history


def predict(theta, X):
    probabilities = sigmoid(np.dot(X, theta))
    predictions = (probabilities >= 0.5).astype(int)

    return predictions


if __name__ == '__main__':
    data_path = "data.csv"
    data = pd.read_csv(data_path,encoding='gbk')

    data['好瓜'] = data['好瓜'].replace({'是':1,'否':0})

    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values.astype(float)

    utils.plot_scatter_diagram(X,y)

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    lambda_param = 0
    
    theta, J_history = train(X, y, alpha=0.1, lambda_param=lambda_param, num_iters=200)

    utils.plot_loss_history(J_history)

    print("斜率 :", theta[1:])
    print("截距 :", theta[0])

    utils.plot_decision_boundary(X, y, theta)

    


