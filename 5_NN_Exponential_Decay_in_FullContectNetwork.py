#---------------------------
#coding:utf-8
# Modify History: 
#          2018 - 7
# Purpose: 
# 
# --------------------------
# 参数优化 1 - 指数衰减学习率 - 以提高学习的效率
# --------------------------
# 学习率应该怎样设置比较合理？ 学习率太高的容易震荡不收敛。学习率太小则收敛的速度比较慢
# 以下表示的是指数衰减学习率的实现过程

import tensorflow as tf
import numpy as np
#-----------------------
# 初始化过程
# Initialzation 
LEARNING_RATE_BASE   = 0.1  #最初的学习率极限 
LEARNING_RATE_DECAY  = 0.99 #学习率衰减率
LEARNING_RATE_STEPS  = 1    #喂入多少轮数据后，更新一次学习率,轮数可以自己设定200, 或者20000。一般取值 = 总样本数/BATCH_SIZE 
LEARNINGSTEPS  = 400        #训练的轮数/次数

# 运行了几轮BARCH_SIZE的计数器，初始值是0，并标注为不可trainable/训练,仅仅是参数
global_step = tf.Variable(0, trainable = False)

# 定义指数学习下降率 - staircase = True 表示学习率沿着阶梯形下降，= False表示平滑下降， global_step 迭代的轮数
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEPS, LEARNING_RATE_DECAY,staircase = True)

# 定义权值w为常量时,并自己定义代价函数
w = tf.Variable(tf.constant(10,dtype = tf.float32))
costfunction  = tf.square( w + 1)

# BackWards Algorthm 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(costfunction,global_step)

with tf.Session() as sess:
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  
  for i in range(LEARNINGSTEPS):
    sess.run(train_step)

    # 关键参数的处理
    learning_rate_value  = sess.run(learning_rate)
    global_step_val      = sess.run(global_step)
    w_val                = sess.run(w)
    costfunction_val     = sess.run(costfunction)

    print  "After training %s, global step is %f. and w value is %f,learning rate is %f, costfunction is %f" %(i,global_step_val,w_val, learning_rate_value, costfunction_val )

print('-----------------------------------------------')
print("以上代码显示了通过指数衰减学习率降低学习率的过程")
print("-----------------------------------------------")
