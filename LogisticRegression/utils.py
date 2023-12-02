import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus'] = False



def plot_loss_history(J_history):
    plt.plot(J_history)
    plt.title('损失函数收敛情况')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.show()



def plot_scatter_diagram(X,y):
    plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'x', color = 'k', s=20, label = '坏瓜')
    plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=20, label = '好瓜')
    plt.title('数据')
    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.legend(loc='upper left')
    plt.show()


def plot_decision_boundary(X, y, theta):
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], marker = 'o', color = 'g', s=20, label = '好瓜')
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], marker = 'x', color = 'k', s=20, label = '坏瓜')

    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = 1/(1+np.exp(-(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1,  colors='b')

    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.legend(loc='upper left')
    plt.show()