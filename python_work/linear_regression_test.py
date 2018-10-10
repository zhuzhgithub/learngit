#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 22:03:52 2018

@author: zzh
"""
import os
from sklearn import linear_model
import numpy as np
import sys
import os 
import matplotlib.pyplot as plt  
def linearModel(data):
    """
    线性回归模型建模步骤展示
    参数----
    data:DataFrame,建模数据
    """
    features = ["x"]
    labels = ["y"]
    
    #划分训练集和测试集
    trainData = data[:15]
    testData = data[15:]
    
    #产生并训练模型，训练模型需要用到x为特征，y为标签类别，得到的模型名为model
    model = trainModel(trainData, features, labels)
    
    #评价模型效果，建立一个evaluateModel去计算error和score；
    error,score = evaluateModel(model, testData, features, labels)
    
    #图形化模型效果，建立一个visualizeModel
    visualizeModel(model, data, features, labels, error, score)
    
    #接下来第一步，用sklearn训练一个模型；
def trainModel(trainData, features, labels):
    """ 利用训练数据，估计模型参数
    参数----
    trainData : DataFrame,训练数据集，包含特征和标签
    labels : 特征名列表
    返回
    ----model : LinearRegressionn, 训练好的线性模型
    """
    #创建一个线性回归模型
    model = linear_model.LinearRegression()

    #训练模型，估计模型参数
    model.fit(trainData[features],trainData[labels])
    return model
   
    #第二步，评价模型；
def evaluateModel(model, testData, features, labels):

    """计算线性模型的均方差和决定系数
    参数----
    model : LinearRegression,训练玩彻骨的线性模型；
    testData : DataFrame,测试数据集；
    features : list[str], 特征名列表；
    labels : list[str], 标签名列表；
    返回----
    error : np.float64， 均方差；
    score : np.float64, 决定系数；
    """
    #均方差（The mean squared error）,均方差越小越好；
    #这里用model去预测testData里面的features，得到的结果减去labels的值，取他们差的平方
    error = np.mean((model.predict(testData[features])-testData[labels])**2)
    
    #决定系数（coefficient of determination）, 决定系数越接近1越好；
    score = model.score(testData[features], testData[labels])
    return error, score


def visualizeModel(model, data, features, labels, error, score):
    """模型可视化"""
    # 在matplotlib中显示中文，设置特殊字体；
    plt.rcParams['font.sans-serif'] = ['SimHei']
    #在图形中创建一个图形框
    fig = plt.figure(figsize = (6,6), dpi = 80)
    #在图形框里只画一幅画
    ax = fig.add_subplot(111)
    #在matplotlib中显示中文，需要使用unicode
    #在python3中，str不需要使用decode
    if sys.version_info[0] == 3:
        ax.set_title(u'%s' % "example of linear_Regression")
    else:
        ax.set_title(u'%s' % "Example of Linear Regression".decode("utf-8"))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    
    #画点图，用蓝色圆点表示原始数据；
    ax.scatter(data[features],data[labels],color = 'b', label = u'%s:$y = x + \epsilon$' % "True Value".decode("utf-8"))
    
    #根据截距的正负，打印不同的标签
    #画线图，用红色的线表示模型结果
    if model.intercept_ > 0:
        ax.plot(data[features], model.predict(data[features]), color='r',
                label=u'%s: $y = %.3fx$ + %.3f'\
                % ("Predcitor".decode("utf-8"), model.coef_, model.intercept_))
    else:
        ax.plot(data[features], model.predict(data[features]), color='r',
                label=u'%s: $y = %.3fx$ - %.3f'\
                % ("Predictor".decode("utf-8"), model.coef_, abs(model.intercept_)))
    legend = plt.legend(shadow = True)
    legend.get_frame().set_facecolor('#6F93AE')
    
    #显示均方差和决定系数；
    ax.text(0.99, 0.01, 
            u'%s%.3f\n%s%.3f'\
            % ("Standard Deviation:".decode("utf-8"), error, "Coefficient of Determination:".decode("utf-8"), score),
            style='italic', verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, color='m', fontsize=13)
    plt.show()
 
    
import pandas as pd   
def readData(path):
    """使用pandas读取数据"""
    data = pd.read_csv(path)
    return data


if __name__ == "__main__":
    homePath = os.path.dirname(os.path.abspath(__file__))
    dataPath = "%s/The_science_of data/intro_ds-master/ch04-linear/simple_example/data/simple_example.csv" % homePath
    data = readData(dataPath)
    linearModel(data)    