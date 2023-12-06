import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LDA:
    def __init__(self):
        self.class_means = None
        self.S_W = None
        self.S_B = None
        self.projection_matrix = None

    def compute_class_means(self, X, y):
        unique_classes = np.unique(y)
        means = [np.mean(X[y == c], axis=0) for c in unique_classes]
        self.class_means = np.array(means)
        return self.class_means

    def compute_within_class_scatter_matrix(self, X, y):
        unique_classes = np.unique(y)
        self.S_W = sum(np.cov(X[y == c], rowvar=False) for c in unique_classes)
        return self.S_W

    def compute_between_class_scatter_matrix(self, X, y):
        overall_mean = np.mean(X, axis=0)
        self.S_B = sum(len(X[y == c]) * np.outer((mean - overall_mean), (mean - overall_mean))
                       for c, mean in zip(np.unique(y), self.class_means))
        return self.S_B

    def compute_lda_projection_matrix(self):
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(self.S_W).dot(self.S_B))
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.projection_matrix = eigenvectors[:, sorted_indices]
        return self.projection_matrix

    def lda_transform(self, X, num_components):
        return np.dot(X, self.projection_matrix[:, :num_components])

    def fit_transform(self, X, y, num_components):
        self.compute_class_means(X, y)
        self.compute_within_class_scatter_matrix(X, y)
        self.compute_between_class_scatter_matrix(X, y)
        self.compute_lda_projection_matrix()
        transformed_data = self.lda_transform(X, num_components)
        return transformed_data, self.projection_matrix

if __name__ == '__main__':
    data_path = "data.csv"
    data = pd.read_csv(data_path, encoding='gbk')
    data['好瓜'] = data['好瓜'].replace({'是': 1, '否': 0})
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values.astype(float)

    lda_model = LDA()
    X_lda, projection_matrix = lda_model.fit_transform(X, y, num_components=1)
    
    w = lda_model.projection_matrix[:, 0]

    plt.rcParams['font.sans-serif']=['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=20, label='好瓜')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='x', color='k', s=20, label='坏瓜')
    plt.plot([-0.3, 0.9], [-0.3 * w[1] / w[0], 0.9 * w[1] / w[0]], color='gray', label='LDA线')

    
    good_proj_points = []
    bad_proj_points = []
    
    for i in range(len(X)):
        x_proj = np.dot(X[i], w) / np.linalg.norm(w)**2 * w
        if y[i] == 0:
            plt.plot([X[i, 0], x_proj[0]], [X[i, 1], x_proj[1]], linestyle='--', color='red', linewidth=0.5)
            bad_proj_points.append(x_proj)
        else:
            plt.plot([X[i, 0], x_proj[0]], [X[i, 1], x_proj[1]], linestyle='--', color='blue', linewidth=0.5)
            good_proj_points.append(x_proj)

    
    good_proj_center = np.mean(np.array(good_proj_points), axis=0)
    bad_proj_center = np.mean(np.array(bad_proj_points), axis=0)
    
    plt.scatter(good_proj_center[0], good_proj_center[1], marker='*', color='g', s=50, label='投影点-好瓜')
    plt.scatter(bad_proj_center[0], bad_proj_center[1], marker='*', color='k', s=50, label='投影点-坏瓜')

    plt.xlabel('密度')
    plt.ylabel('含糖量')
    plt.title('原始特征空间中的LDA决策边界线')
    plt.xlim(-0.3,0.9)
    plt.ylim(-0.3,0.6)
    plt.legend()
    plt.show()




    