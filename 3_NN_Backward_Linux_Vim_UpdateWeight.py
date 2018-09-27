#-------------------------
#coding:utf-8
# Modify History: 
#          2018 - 7
# Purpose:
#     1. 设定损失函数loss  = (w - 1) ^2  ，并设定w 初始化成常数 5。反向传播就是求出最优解w,即1求出loss最小时对应的w值
#     2. 学习率大时，会使得迭代过程中震荡比较大。学习率小时，需要迭代的次数较多
#     

#-----------------------
import tensorflow as tf
learningrate  = 0.1
TRAIN_STEPS   = 200
# -------------
# 定义待优化的参数w 为常量5 
# w initialize to constant 5 and dtype  = tf.float32 ) 
w  = tf.Variable(tf.constant(1005, dtype = tf.float32))

# ------------------
# cost function - 
# 代价函数设定为 L = ( w  - 1 )^2 
costfunction  = tf.square(w -  1) 

#---------------------
# Backwoard Algorithm 
#-------------------
train_step    = tf.train.GradientDescentOptimizer(learningrate).minimize(costfunction )
# Session Runing the result 
with tf.Session() as sess:
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  for i in range(TRAIN_STEPS): # 训练200次
    sess.run(train_step)
    w_val    = sess.run(w)
    cost_val = sess.run(costfunction)
    # 找到最优的w  = 1  
    print "After %d train steps ,opt w value is %f and costfunction_value is %f" %(i, w_val,cost_val)
print ('-----------------------')
print "Training Completed !!! "
print ('----------------------')
print ("以上验证了反向传播是为了找到是代价函数最小时对应的权值w1")
# --------------------------
