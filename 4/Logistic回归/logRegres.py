'''
可看做概率估计
Logistic回归的一般过程 :
(1) 收集数据：采用任意方法收集数据。
(2) 准备数据：由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
(3) 分析数据：采用任意方法对数据进行分析。
(4) 训练算法：大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
(5) 测试算法：一旦训练步驟完成，分类将会很快。
(6) 使用算法：首先，我们需要输入一些数据，并将其转换成对应的结构化数值；接着，基于训练好的回归系数就可以对这些数值进行简单的回归计算，判定它们属于哪个类别 . ，在这之后， 我们就可以夺输出的类别上做一些其他分析工作。
优点：计算代价不高，易于理解和实现
缺点：容易欠拟合，分类精度可能不高
适用数据类型：数值型和标称型
'''
import numpy as np
from numpy import*
import matplotlib
import matplotlib.pyplot as plt
import math
'''
1.梯度上升法伪代码：
每个回归系数初始化为 1
重复R次：
    计算整个数据集的梯度
    使用 alpha x gradient更新回归系数的向量
    返回回归系数
2.随机梯度上升伪代码：
所有回归系数初始化为 1
对数据集中每个样本
    计算该样本的梯度
    使用 alpha x gradient 更新回归系数值
返回回归系数值
'''

#======================项目案例1: 使用 Logistic 回归在简单数据集上的分类===================

# 数据集
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  #逐行读入并切分，每行的前两个值为X1，X2
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])   ##X0设为1.0，保存X1，X2;加上了特征x0,值为1，成了100*3的矩阵
        labelMat.append(int(lineArr[2]))    #每行第三个值对应类别标签
    return dataMat,labelMat

# S 函数
def sigmoid(inX):
    return 1.0/(1 + exp(-inX))     #RuntimeWarning: overflow encountered in exp ;错误提示原因是sigmoid函数溢出


# Logistic回归梯度上升优化算法
def gradAscent(dataMatIn,classLabels):
    # dataMatIn 是一个2维NumPy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。它包含了两个特征x1和x2 ，再加上第 0维特征x0，所以dataMatIn存放的将是 100x3 的矩阵。
    # classLabels 是类别标签，它是一个 1*100 的行向量。为了便于矩阵计算，需要将该行向量转换为列向量，做法是将原向量转置，再将它赋值给labelMat。
    dataMatrix = mat(dataMatIn)   # 转化为矩阵[[1,1,2],[1,1,2]....]
    labelMat = mat(classLabels).transpose()   # 首先将数组转换为 NumPy 矩阵，然后再将行向量转置为列向量
    m,n = shape(dataMatrix)     # m->数据量，样本数； n->特征数
    alpha = 0.001     # alpha代表向目标移动的步长
    maxCycles = 500   #迭代次数
    weights = ones((n,1))    # weights 代表回归系数， 此处的 ones((n,1)) 创建一个长度和特征数相同的矩阵，其中的数全部都是 1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)    # 矩阵乘法：(m x n) * (n*1) = (m*1)
        error = (labelMat - h)   # 向量相减，labelMat是实际值,h是UI个列向量，元素个数是样本数100
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights


#随机梯度上升算法
# 随机梯度上升算法与梯度上升算法在代码上很相似，但也有一些区别: 第一，后者的变量 h 和误差 error 都是向量，而前者则全是数值；第二，前者没有矩阵的转换过程，所有变量的数据类型都是 NumPy 数组
def stocGradAscent0(dataMatrix,classLabels):
    #m,n = shape(dataMatrix)    #都是矩阵，不是数组
    #alpha = 0.01
    #weights = ones(n)   # Logistic回归梯度上升优化算法
    dataMatrix = np.array(dataMatrix)  # 把处理后的列表转化为数组
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))   # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,此处求出的 h 是一个具体的数值，而不是一个矩阵
        error = classLabels[i]-h     # 计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
        weights = weights + alpha * error *dataMatrix[i]    # 0.01*(1*1)*(1*n)
    weights = np.mat(weights).reshape((3, 1))   ##这个目的是方便下面plotBestFit()函数中,wei.getA()的操作，其中的wei只能为矩阵
    return weights


#改进的随机梯度上升算法(随机化)
# 第一处改进为 alpha 的值。alpha 在每次迭代的时候都会调整，这回缓解上面波动图的数据波动或者高频波动。另外，虽然 alpha 会随着迭代次数不断减少，但永远不会减小到 0。
# 第二处修改为 randIndex 更新，这里通过随机选取样本拉来更新回归系数。这种方法将减少周期性的波动。这种方法每次随机从列表中选出一个值，然后从列表中删掉该值（再进行下一次迭代）。
def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)    # 创建与列数相同的矩阵的系数矩阵，所有的元素都是1
    for j in range(numIter):    # 随机梯度, 循环150,观察是否收敛
        dataIndex = list(range(m))     # [0, 1, 2 .. m-1]
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01   # i和j的不断增大，导致alpha的值不断减少，但是不为0
            randIndex = int(random.uniform(0,len(dataIndex)))    # 随机产生一个 0～len()之间的一个值；random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            h = sigmoid(sum(dataMatrix[randIndex]*weights))      # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
            error = classLabels[randIndex] - h
            weights = weights + alpha * error *dataMatrix[randIndex]
            del(dataIndex[randIndex])
    weights = np.mat(weights).reshape((n, 1))   #将stocGradAscent0中的weights = np.mat(weights).reshape((3, 1))，其中的3改为n即可
    return weights

#分析数据：画出决策边界
def plotBestFit(wei):
    weights = wei.getA()  #getA()函数：将numpy矩阵转换为数组
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []   #xcord1,ycord1代表正例特征
    ycord1 = []
    xcord2 = []   #xcord2,ycord2代表负例特征
    ycord2 = []
    for i in range(n):    #循环筛选出正负集
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1 = ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    p2 = ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)    #设定边界直线x和y的值
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    ax.legend((p1, p2), ('class1', 'class0'), loc=2)
    plt.show()


#=====================项目案例2：从疝气病症预测病马的死亡率==============================
'''
1.项目概述
使用 Logistic 回归来预测患有疝病的马的存活问题。疝病是描述马胃肠痛的术语。然而，这种病不一定源自马的胃肠问题，其他问题也可能引发马疝病。这个数据集中包含了医院检测马疝病的一些指标，有的指标比较主观，有的指标难以测量，例如马的疼痛级别。

2.开发流程
收集数据: 给定数据文件
准备数据: 用 Python 解析文本文件并填充缺失值
分析数据: 可视化并观察数据
训练算法: 使用优化算法，找到最佳的系数
测试算法: 为了量化回归的效果，需要观察错误率。根据错误率决定是否回退到训练阶段，
         通过改变迭代的次数和步长的参数来得到更好的回归系数
使用算法: 实现一个简单的命令行程序来手机马的症状并输出预测结果并非难事，
         这可以作为留给大家的一道习题
***处理缺失值的做法：
□ 使用可用特征的均值来填补缺失值；
□ 使用特殊值来填补缺失值，如 -1;
□ 忽略有缺失值的样本；
□ 使用相似样本的均值添补缺失值；
□ 使用另外的机器学习算法预测缺失值。
'''
#准备数据：处理数据中的缺失值；使用0代替缺失值不会对结果有影响；数据已处理
#可以通过处理后的数据集和Logistic优化算法预测病马的生死问题

#测试算法：用Logistic回归进行分类
#Logistic回归分类函数
def classifyVector(inX,weights):  #输入参数为：inX -- 特征向量，weights -- 根据梯度下降/随机梯度下降计算得到的回归系数
    prob = sigmoid(sum(inX*weights))   # S 函数结果
    if prob>0.5:
        return 1.0
    else:
        return 0.0

#打开测试集和训练集，并对数据进行格式化处理
def colicTest():
    frTrain = open('horseColictraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []    # trainingSet 中存储训练数据集的特征
    trainingLabels = []   #trainingLabels 存储训练数据集的样本对应的分类标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):    #数据集中一共有21列特征
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)   # 使用改进后的随机梯度下降算法求得在此数据集上的最佳回归系数 trainWeights
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():   # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print('the error rate of this test is :%f' % errorRate)
    return errorRate

# 调用 colicTest() 10次并求结果的平均值
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print('\n after %d iteration the average error rate is:%f ' % (numTests,errorSum/float(numTests)))



if __name__ == '__main__':
    # (a)梯度下降算法迭代500次。(b)随机梯度下降算法迭代200次。 (c)改进的随机梯度下降算法迭代20次。(d)改进的随机梯度下降算法迭代200次。
    # 测试梯度上升优化函数
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    print('the first weights is: \n', weights)
    plotBestFit(weights)       # 画出决策边界
    #测试随机梯度上升算法
    weights1 = stocGradAscent0(array(dataArr), labelMat)
    print('the second weights is: \n', weights1)
    plotBestFit(weights1)
    # 测试随机梯度上升算法（随机化）
    weights2 = stocGradAscent1(array(dataArr), labelMat)
    print('the third weights is: \n', weights2)
    plotBestFit(weights2)
    # 测试随机梯度上升算法（随机化）：修改默认迭代次数
    weights22 = stocGradAscent1(array(dataArr), labelMat, 500)
    print('the fourth weights is: \n', weights22)
    plotBestFit(weights22)
    #测试分类
    multiTest()



