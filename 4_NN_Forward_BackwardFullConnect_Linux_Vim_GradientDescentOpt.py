#----------------------------------
#coding:utf-8
#
# Modify History: 
#          2018 - 7
# Purpose: 
#        通过前向传播算法来搭建前向传播算法+ 并通过session.run 来执行计算
#        通过反向传播来训练模型参数，并在所有参数梯度上使用梯度下降方法，使得NN在所有训练数据上的损失函数最小
#    
#  神经网络的实现过程：
#  1. 准备数据集合及提取特征，作为输入喂给NN 
#  2. 通过前向传播过程打架NN,并由session.run 执行计算结果
#  3. 将特征送入NN，通过后向传播算法来更新权重以寻找使得代价函数最小的权重
#  4. 使用训练好的模型做预测和测试
#         
# Platform: 
#         Ubuntu 16.04 + Python2.7 + CPU + Tensorflow Version2 
#  
# 导入数据模块，生成模拟数据集和
import tensorflow as tf
import numpy as np 

# -----------------------------
# 常量变量声明
BATCH_SIZE    = 8*2 # 每次喂入神经网络的数据大小
seed          = 23455 # 
LearningRate  = 0.001 # 每次参数迭代的过程的

# 基于seed 产生随机数
rng = np.random.RandomState(seed)
# input paramters  - M*N = Example * Features
# 随机数返回32×2 的矩阵。32表示32组数，2 表示两个特征： 重量和体积 作为输入数据集 
X   = rng.rand(32,2) # 2 Features , 32  = Examples 

# 人为为数据加上标签，认为体积和重量之和<1的认为是1，标记为合格。 体积和重量>1的，认为是0标记为不合格。
Y   = [ [ int ( x0 + x1 < 1) ] for(x0,x1) in X]
print "X: \n", X  # 数据集X
print "Y: \n", Y  # 标签Y
#print "This is type() = ", type(Y)  

# 1 定义神经网络的输入，参数和输出，定义前向传播过程 
x   = tf.placeholder(tf.float32, shape = (None,2)) # 2 Features，组数不知道，写None 
y_  = tf.placeholder(tf.float32, shape = (None,1)) # y_表示标准答案，合格是1不合格是0的标签；每个标签是合格或不合格，就一个元素。行数不确定写None ,列数写1（合格或不合格2选一）

w1  = tf.Variable(tf.random_normal([2,3], stddev = 1, seed = 1))  # 2*3 表示输入和隐藏层的
w2  = tf.Variable(tf.random_normal([3,1], stddev = 1, seed = 1))  # 一个输出结果，列数写1

#----------------------------------------------------
# Neural Network Output Variable -  Forward Algorithm
#----------------------------------------------------   
a   = tf.matmul(x,w1)
y   = tf.matmul(a,w2)
#print('------------------------')
#print('y_ = ', y_)

#---------------------------------------------
# 2 Define CostFunction and Bacward Algorithm
# 反向传播通过更新权重参数，使代价函数最小以寻找到最优解 
#---------------------------------------------
costfunction  = tf.reduce_mean(tf.square(y - y_))  # MMSE function 来定义代价函数

# 迭代模型
train_step    = tf.train.GradientDescentOptimizer(LearningRate).minimize(costfunction) #通过梯度下降法来表示训练过程
#train_step = tf.train.MomentumOptimizer(LearningRate,0.09).minimize(costfunction)
#train_step = tf.train.AdamOptimizer(LearningRate).minimize(costfunction)

# 3 Session - Run Result 
with tf.Session() as sess:
    # Initialization Parameters - 初始化参数
    init_op = tf.global_variables_initializer() # 初始化所有的参数
    sess.run(init_op)
    print '未优化前的参数和数值:'
    print "w1 = : \n", sess.run(w1)  # 打印出优化前的参数
    print "w2 = : \n", sess.run(w2)  # 打印出优化前的参数
    
    # Training Module ----  训练过程
    STEPS  =  10000 # 训练轮
    for i in range(STEPS): 
      start = (i*BATCH_SIZE) % 32   # start~end 0-8 8-16 16-24  24-32
      end   = start + BATCH_SIZE

      # 在数据和标签中，取出特定BATCH_SIZE 长度的数据喂入神经网络
      sess.run(train_step, feed_dict = {x: X[start:end], y_: Y[start:end] } ) # 将随机数据集和标签集中的start to end 数据喂入神经网络，并有sess.run来执行训练过程
      if i % 500 == 0:
        total_loss = sess.run(costfunction, feed_dict = {x: X, y_:Y})
        print("After %d traning steps ,totalloss on all data is %g" % (i,total_loss))
    

# 4. 输出训练后的参数值
    print('训练后的参数值:')
    print"w1 = \n",sess.run(w1)
    print"w2 = \n",sess.run(w2)
   




































