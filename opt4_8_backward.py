#coding:utf-8
#神经网络搭建的八股之三：opt4_8_backward.py
#------------------------------------------------
# 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import opt4_8_generateds
import opt4_8_forward

STEPS      = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE  = 0.01 # 指数衰减学习率初始值
LEARNING_RATE_DECAY = 0.99  # 指数衰减学习率
REGULARIZER         = 0.01 

def backward():
  x  = tf.placeholder(tf.float32,shape = (None, 2))
  y_ = tf.placeholder(tf.float32,shape = (None, 1))
  
  # 生成数据集
  X,Y_,Y_c  = opt4_8_generateds.generateds()
  
  # 前向传播
  y = opt4_8_forward.forward(x,REGULARIZER)
  
  # 迭代基数器，定义为不可训练 - 指数衰减学习率
  global_step  = tf.Variable(0, trainable = False)
  learning_rate  = tf.train.exponential_decay(
      LEARNING_RATE_BASE,
      global_step,
      300/BATCH_SIZE,
      LEARNING_RATE_DECAY, 
      staircase = True) 

  # 正则化之后的代价函数
  loss_mse   = tf.reduce_mean(tf.square(y - y_))
  loss_total = loss_mse  + tf.add_n(tf.get_collection('losses') ) # 正则化之后所有权重w代价函数的和

  # 定义反向传播算法 - 包含正则化
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)
  
  with tf.Session() as sess: 
      init_op  = tf.global_variables_initializer() 
      sess.run(init_op)
      for i in range(STEPS):  
         start  = (i* BATCH_SIZE) % 300 
         end    = start + BATCH_SIZE 
         sess.run(train_step,feed_dict = {x:X[start:end],  y_:Y_[start:end]})
         if i%2000 == 0:  
            loss_value  = sess.run(loss_total, feed_dict  = {x:X,  y_:Y_ })
            print ("After %d train steps, loss is %f :" % (i, loss_value ))
      
      xx,yy = np.mgrid[-3:3:0.01, -3:3:0.01 ]
      grid  = np.c_[xx.ravel(), yy.ravel()]
      probs = sess.run(y, feed_dict={x: grid})
      probs = probs.reshape(xx.shape )

  plt.scatter( X[:,0], X[:,1], c= np.squeeze(Y_c) )
  plt.contour( xx,yy,probs,levels = [0.5] )
  plt.show()

# -------------
if __name__ == '__main__':
    backward()










