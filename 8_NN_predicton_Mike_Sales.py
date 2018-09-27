#-------------------
#coding:utf-8
#
# Modify History: 
#          2018 - 7
# Purpose: 
#    1.搭建NN框架，并通过自定义代价函数，实现酸奶的预测
#    2.酸奶成本1元，酸奶利润9元; 预测的少了不够买的，损失9×M元（M少买的数量）所以模型应该往多预测的方向发展
#

# 导入模块生成数据集合
import tensorflow as tf
import numpy as np 
BATCH_SIZE = 8
SEED       = 23455
LEARNINGRATE  = 0.001
COST       = 1
PROFIT     = 9
# Initialization for the Data 
rdm  = np.random.RandomState(SEED)
X    = rdm.rand(32,2)  # 输32×2 的输入数据集合

# ----------
#自己拟成数据集合 Y_ = x1 + x2 + (-0.05 to +0.05 ) 作为标准答案（标签）
# ---------
# 这里rdm.rand() 生成 0-1 之间的随机数。/10 后变成0-0.1; -0.5后变成 -0.05 -+ 0.05 之间的随机噪声
# Y_  Here randn()/10 - 0.05 = -0.05 ~ + 0.05 
Y_   = [[x1 + x2 + (rdm.rand()/10 - 0.05)] for (x1,x2) in X]

#1 Define the inputs and outputs , forward Algorithm 
x  = tf.placeholder(tf.float32,shape = (None, 2))
y_ = tf.placeholder(tf.float32,shape  = (None,1))
w1 = tf.Variable(tf.random_normal([2,1], stddev = 1, seed  = 1))
y  = tf.matmul(x,w1)


# 2 CostFunction and Back Algorithm 
#loss  = tf.reduce_mean(tf.square(y - y_))   - MMSE 函数构建代价函数loss 
#----------------------------------
# 通过自定义函数来定义代价函数,并由梯度下降法来做迭代
# tf.where(tf.greater(a,b),1,0) ) # 如果a>b 则结果是1，否则是0
loss  = tf.reduce_sum(tf.where(tf.greater(y_,y), PROFIT*(y_ - y), COST*(y - y_) )) # 自定义代价函数
train_step = tf.train.GradientDescentOptimizer(LEARNINGRATE).minimize(loss)


# 3 Session , Traning Data 
with tf.Session() as sess: 
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print'Start Training ...'
    STEPS  = 20000
    for i in range(STEPS):
      start  = (i * BATCH_SIZE) % 32
      end    = start + BATCH_SIZE
      sess.run(train_step, feed_dict  = {x:X[start:end], y_: Y_[start:end]  })
      
      if i % 500 == 0 :
        totallost = sess.run(loss,feed_dict = {x:X, y_:Y_})
        print "After %d traning steps ,w1 is : "% i
        print sess.run(w1), "\n"
        
    print "Final w1 of train is :\n", sess.run(w1)
    
