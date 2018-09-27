# -----------------------
#coding:utf-8
# Modify History: 
#          2018 - 7
# Purpose: 
#        两层简单神经网络（全链接) - 搭建神经网络及前向传播算法的实现过程
# Platform: 
#         Ubuntu 16.04 + Python2.7 + CPU + Tensorflow Version2 

# 
import tensorflow as tf
# 定义参数和输入变量
# 生成1*2 的常量矩阵
# x  = tf.constant([[0.7, 0.5]])
# -------------------------------------
# 这里是为了为输入x占位，shape = 2 ,表示将多组数据的两个输入特征来喂入神经网络
x  =tf.placeholder(tf.float32,shape  = (None,2)) # 2表示有几个输入特征值

# 生成2×3 方差 =1 ，随机种子 = 1的正太分布
#  随机生成输入层到隐藏层的第一个权值w1 
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1, seed = 1))

# 随机生成隐藏层到输出层的取值w2
# 生成3×1 方差 =1 ，随机种子 = 1的正太分布
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))

# 定义前向传播算法 - 搭建计算网络并由会话层来执行计算过程
a = tf.matmul(x,w1) # 矩阵乘法
y = tf.matmul(a,w2) # 矩阵乘法
#  -----------------
# Using Seesing to run the resul

# 使用会话计算结果 
with tf.Session() as sess:
    # 将初始化所有变量的函数简写成初始化节点 - 所有变量的初始化
    init_op = tf.global_variables_initializer()
    
    sess.run(init_op)
    #print "y in tf3_3.py is: \n",sess.run(y) # used to x= tf.constant([[1.2,2.3]])
    
    # 通过字典dict喂入神经网络4组数据，每个数据带有两个feature 
    print"y in tf3_3.py is:\n", sess.run(y,feed_dict = {x: [[0.7, 0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}) # 这里的列都是2表示输入两个特征（一个表示重量，一个表示体积) ，跟place holder 里面初始定义的列是相同的。。不过是多行。每行表示一个Example
    print "w1 = \n",sess.run(w1)
    print "w2 = \n",sess.run(w2)

print("\n该程序执行了：")
print("两层简单神经网络（全链接) - 搭建神经网络及前向传播算法的实现过程")

'''
y in tf3_3.py is : 
[ [  ?? ]]
'''
