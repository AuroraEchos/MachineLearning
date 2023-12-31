逻辑回归是一种用于分类问题的机器学习算法。它通过将特征与权重的线性组合输入到一个sigmoid函数中，将其转换为概率值，然后根据阈值将数据分为两个类别。


基本步骤：
1.定义模型参数：逻辑回归模型的参数包括权重（w）和偏置（b）。
2.定义激活函数：逻辑回归模型通常使用sigmoid函数作为激活函数，将输出值映射到[0, 1]的范围内。
3.定义损失函数：逻辑回归模型通常使用交叉熵损失函数来衡量预测值与真实值之间的差异。
4.定义优化算法：可以使用梯度下降算法来最小化损失函数，并更新模型参数。
5.进行模型训练：使用训练集数据进行模型训练，迭代更新模型参数。
6.进行模型预测：使用训练好的模型参数对新数据进行预测，通过代入参数和计算激活函数得到预测结果。


编程实现：
1.sigmoid
    实现了 logistic sigmoid 函数，将输入值映射到 (0, 1) 之间。
2.cost_function
    计算逻辑回归的损失函数和梯度。损失函数使用交叉熵损失函数来衡量预测值与实际值之间的差异。
3.gradient_descent
    使用梯度下降算法来更新模型参数 theta，以最小化损失函数。
4.train
    初始化模型参数 theta 并调用梯度下降函数进行训练，返回训练得到的参数和损失函数的历史记录。
5.predict
    使用训练好的参数进行预测，根据阈值 0.5 将概率值转换为二分类的预测结果。


