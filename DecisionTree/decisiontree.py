import numpy as np
import graphviz

class Node:
    def __init__(self, attribute=None, value=None, label=None):
        self.attribute = attribute
        self.value = value
        self.label = label
        self.children = {}

# 信息熵
def calculate_entropy(data):
    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return entropy

# 信息增益
def calculate_information_gain(data, attribute_index):
    attribute_values = data[:, attribute_index]
    unique_values = np.unique(attribute_values)
    entropy = calculate_entropy(data)
    gain = entropy

    for value in unique_values:
        subset = data[data[:, attribute_index] == value]
        subset_entropy = calculate_entropy(subset)
        subset_prob = len(subset) / len(data)
        gain -= subset_prob * subset_entropy
        
    return gain

# 根据信息增益选择最佳划分属性
def choose_best_attribute(data, attributes):
    num_attributes = data.shape[1] - 1
    gains = np.zeros(num_attributes)

    for i in range(num_attributes):
        gains[i] = calculate_information_gain(data, i)

    best_attribute_index = np.argmax(gains)
    
    return best_attribute_index

# 递归构建决策树
def build_decision_tree(data, attributes):
    labels = data[:, -1]
    
    if len(np.unique(labels)) == 1:
        return Node(label=labels[0])
    
    if data.shape[1] == 1:
        unique_labels, counts = np.unique(labels, return_counts=True)
        label = unique_labels[np.argmax(counts)]
        return Node(label=label)

    best_attribute_index = choose_best_attribute(data, attributes)
    best_attribute = attributes[best_attribute_index]
    root = Node(attribute=best_attribute)

    attribute_values = np.unique(data[:, best_attribute_index])
    
    for value in attribute_values:
        subset = data[data[:, best_attribute_index] == value]
    
        if len(subset) == 0:
            unique_labels, counts = np.unique(labels, return_counts=True)
            label = unique_labels[np.argmax(counts)]
            root.children[value] = Node(label=label)
        else:
            root.children[value] = build_decision_tree(subset, attributes)

    return root

# 使用构建好的决策树进行预测
def predict(root, sample, attributes):
    if root.label is not None:
        return root.label

    attribute_value = sample[attributes.index(root.attribute)]

    if attribute_value not in root.children:
        return None

    child_node = root.children[attribute_value]

    return predict(child_node, sample, attributes)

# 决策树可视化
def visualize_tree(node, dot=None):
    if dot is None:
        dot = graphviz.Digraph(comment='Decision Tree', graph_attr={'dpi':'300'})

    if node.label is not None:
        dot.node(str(id(node)), label=str(node.label), shape='ellipse', fontname='FangSong')
    else:
        dot.node(str(id(node)), label=str(node.attribute), shape='box', fontname='FangSong')

    for value, child_node in node.children.items():
        if child_node.label is not None:
            dot.node(str(id(child_node)), label=str(child_node.label), shape='ellipse', fontname='FangSong')
        else:
            dot.node(str(id(child_node)), label=str(child_node.attribute), shape='box', fontname='FangSong')
        
        dot.edge(str(id(node)), str(id(child_node)), label=str(value), fontname='FangSong')

        visualize_tree(child_node, dot)

    return dot


data = np.array([
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
    ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
])

attributes = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']

build_decision_tree(data, attributes)


root = build_decision_tree(data, attributes)
dot = visualize_tree(root)
dot.render('decision_tree', format='png', cleanup=True)



sample = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑']
prediction = predict(root=root, sample=sample, attributes=attributes)
print('预测结果: ', prediction)



""" 
以下为各步骤计算结果：
    1、信息熵为 0.998
    2、色泽信息增益 0.109 --[乌黑 0.918]   [浅白 0.721]   [青绿 0.999]
       根蒂信息增益 0.143 --[硬挺 -1.442]  [稍蜷 0.985]   [蜷缩 0.954]
       敲声信息增益 0.141 --[沉闷 0.970]   [浊响 0.970]   [清脆 -1.442]
       纹理信息增益 0.381 --[模糊 -1.442]  [清晰 0.764]   [稍糊 0.721]
       脐部信息增益 0.289 --[凹陷 0.863]   [平坦 -1.442]  [稍凹 0.999]
       触感信息增益 0.006 --[硬滑 0.999]   [软粘 0.970]
    3、选出纹理为划分属性
"""