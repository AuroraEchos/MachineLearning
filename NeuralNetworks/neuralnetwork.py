import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def preprocess(data):
    for title in data.columns:
        if data[title].dtype == 'object':
            encoder = LabelEncoder()
            data[title] = encoder.fit_transform(data[title])
    ss = StandardScaler()
    X = ss.fit_transform(data.drop('好瓜', axis=1))
    x, y = np.array(X), np.array(data['好瓜']).reshape(-1, 1)
    return x, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def plot_loss_curve(loss_list, xlabel, ylabel, title=''):
    plt.plot(range(1, len(loss_list) + 1), loss_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def standard_BP(x, y, dim=10, eta=0.8, max_iter=500):
    n_samples, n_features = x.shape
    w1, b1 = np.zeros((n_features, dim)), np.zeros((1, dim))
    w2, b2 = np.zeros((dim, 1)), np.zeros((1, 1))
    loss_list = []

    for ite in range(max_iter):
        loss_per_ite = []

        for m in range(n_samples):
            xi, yi = x[m, :], y[m, :]
            xi, yi = xi.reshape(1, -1), yi.reshape(1, -1)

            u1 = np.dot(xi, w1) + b1
            out1 = sigmoid(u1)
            u2 = np.dot(out1, w2) + b2
            out2 = sigmoid(u2)

            loss = np.square(yi - out2) / 2
            loss_per_ite.append(loss.item())
            print('iter: {}  loss: {:.4f}'.format(ite, loss.item()))

            d_out2 = -(yi - out2)
            d_u2 = d_out2 * d_sigmoid(out2)
            d_w2 = np.dot(out1.T, d_u2)
            d_b2 = d_u2

            d_out1 = np.dot(d_u2, w2.T)
            d_u1 = d_out1 * d_sigmoid(out1)
            d_w1 = np.dot(xi.T, d_u1)
            d_b1 = d_u1

            w1, w2 = w1 - eta * d_w1, w2 - eta * d_w2
            b1, b2 = b1 - eta * d_b1, b2 - eta * d_b2

        loss_list.append(np.mean(loss_per_ite))

    plot_loss_curve(loss_list, '迭代次数', '损失', '标准 BP')

    return w1, w2, b1, b2


def accumulate_BP(x, y, dim=10, eta=0.8, max_iter=500):
    n_samples, n_features = x.shape
    w1, b1 = np.zeros((n_features, dim)), np.zeros((n_samples, dim))
    w2, b2 = np.zeros((dim, 1)), np.zeros((n_samples, 1))
    loss_list = []

    for ite in range(max_iter):
        u1 = np.dot(x, w1) + b1
        out1 = sigmoid(u1)
        u2 = np.dot(out1, w2) + b2
        out2 = sigmoid(u2)

        loss = np.mean(np.square(y - out2)) / 2
        loss_list.append(loss)
        print('iter: {}  loss: {:.4f}'.format(ite, loss))

        d_out2 = -(y - out2)
        d_u2 = d_out2 * d_sigmoid(out2)
        d_w2 = np.dot(out1.T, d_u2)
        d_b2 = d_u2

        d_out1 = np.dot(d_u2, w2.T)
        d_u1 = d_out1 * d_sigmoid(out1)
        d_w1 = np.dot(x.T, d_u1)
        d_b1 = d_u1

        w1, w2 = w1 - eta * d_w1, w2 - eta * d_w2
        b1, b2 = b1 - eta * d_b1, b2 - eta * d_b2

    plot_loss_curve(loss_list, '迭代次数', '损失', '累积 BP')

    return w1, w2, b1, b2


def main():
    data = pd.read_table('watermelon_3.txt', delimiter=',')
    data.drop('编号', axis=1, inplace=True)
    x, y = preprocess(data)
    dim = 10
    standard_BP(x, y, dim)
    accumulate_BP(x, y, dim)


if __name__ == '__main__':
    main()
