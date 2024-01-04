import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")

# 打印模型评估结果的辅助函数
def print_model_evaluation(file, model_name, scores):
    table = [
        ["准确度", scores['test_accuracy'].mean()],
        ["精确度", scores['test_precision_macro'].mean()],
        ["召回率", scores['test_recall_macro'].mean()],
        ["F1分数", scores['test_f1_macro'].mean()]
    ]
    headers = [model_name, '平均值']
    result_str = tabulate(table, headers, tablefmt="fancy_grid")
    print(result_str)
    
    # 将结果写入文件
    with open(file, 'a') as f:
        f.write(f"\n{model_name} 评估结果:\n")
        f.write(result_str)
        f.write("\n\n")

# 加载并训练鸢尾花数据集
def load_and_train_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    y = pd.Series(iris['target_names'][iris['target']])
    
    # 线性核SVM
    linear_svm = svm.SVC(C=1, kernel='linear')
    linear_scores = cross_validate(linear_svm, X, y, cv=5, scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'), return_train_score=False)
    print_model_evaluation("model_evaluation_results.txt", "鸢尾花 - 线性核SVM", linear_scores)

    # 高斯核SVM
    rbf_svm = svm.SVC(C=1, kernel='rbf')
    rbf_scores = cross_validate(rbf_svm, X, y, cv=5, scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'), return_train_score=False)
    print_model_evaluation("model_evaluation_results.txt", "鸢尾花 - 高斯核SVM", rbf_scores)

    # BP神经网络
    bp_nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    bp_scores = cross_validate(bp_nn, X, y, cv=5, scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'), return_train_score=False)
    print_model_evaluation("model_evaluation_results.txt", "鸢尾花 - BP神经网络", bp_scores)

    # 决策树
    tree_classifier = DecisionTreeClassifier()
    tree_scores = cross_validate(tree_classifier, X, y, cv=5, scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'), return_train_score=False)
    print_model_evaluation("model_evaluation_results.txt", "鸢尾花 - 决策树", tree_scores)

# 加载并训练乳腺癌数据集
def load_and_train_breast_cancer():
    breast_cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(breast_cancer['data'], columns=breast_cancer['feature_names'])
    y = pd.Series(breast_cancer['target_names'][breast_cancer['target']])

    # 线性核SVM
    linear_svm = svm.SVC(C=1, kernel='linear')
    linear_scores = cross_validate(linear_svm, X, y, cv=5, scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'), return_train_score=False)
    print_model_evaluation("model_evaluation_results.txt", "乳腺癌 - 线性核SVM", linear_scores)

    # 高斯核SVM
    rbf_svm = svm.SVC(C=1, kernel='rbf')
    rbf_scores = cross_validate(rbf_svm, X, y, cv=5, scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'), return_train_score=False)
    print_model_evaluation("model_evaluation_results.txt", "乳腺癌 - 高斯核SVM", rbf_scores)

    # BP神经网络
    bp_nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    bp_scores = cross_validate(bp_nn, X, y, cv=5, scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'), return_train_score=False)
    print_model_evaluation("model_evaluation_results.txt", "乳腺癌 - BP神经网络", bp_scores)

    # 决策树
    tree_classifier = DecisionTreeClassifier()
    tree_scores = cross_validate(tree_classifier, X, y, cv=5, scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'), return_train_score=False)
    print_model_evaluation("model_evaluation_results.txt", "乳腺癌 - 决策树", tree_scores)

# 主函数
def main():
    load_and_train_iris()
    load_and_train_breast_cancer()

if __name__ == '__main__':
    main()
