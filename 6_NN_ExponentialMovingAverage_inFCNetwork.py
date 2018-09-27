#----------------------
#coding:utf-8
# -----------------------
# 参数优化2 - 滑动平均
# -----------------------
# 滑动平均（影子值）记录了每个参数一段时间内过往值的平均。增加了模型的泛化性。
# 针对所有参数w 和b - 像是给参数加了影子，参数变化影子缓慢追谁
# 影子 = 衰减率 × 影子 + （1 - 衰减率）× 参数  影子初值 = 参数初值
# 衰减率 = min{MOVING_AVERAGE_DECAT(滑动平均衰减率一般值比较大，取0.99), 1 + 轮数/10 + 轮数}
#
import tensorflow as tf 
# 
# 参数初始化 
w1 = tf.Variable(0,dtype = tf.float32)
global_step = tf.Variable(0,trainable = False)
#-------------------------------------------------
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率一般较大
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

#----------------------------------------
# 对所有待训练的参数求滑动平均并训练成列表
ema_op = ema.apply(tf.trainable_variables())


# 工程中通常将训练过程和求滑动平均结合起来使用 
# with  tf.control_ dependencies([train_steps,ema_op])
    # train_op  = tf.no_op(name = 'train')

#-------------------------------------------
with tf.Session() as sess:
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  
  # ema.average(w1) - calcualted the average value of w1
  print ("第一次求解滑动平均：")
  print sess.run([w1,ema.average(w1)])  # ema.average(w1) 计算滑动平均参数值
  
  # 为参数值重新定复制为1 
  print("w1的值被重新赋值成1以后的滑动值：")
  sess.run(tf.assign(w1,1))
  sess.run(ema_op)
  print sess.run([w1, ema.average(w1)])
 
  # 更新step and w1的值，模拟出100轮迭代后，参数w1变为10 
  # print "train step  = %f, w1 is = %f "  %(global_step, w1)
  print "train step is 100 and w1 parameter re-assign to 10 "
  sess.run(tf.assign(global_step,100))
  sess.run(tf.assign(w1, 10))
  sess.run(ema_op) # execulate the node 
  print sess.run([w1, ema.average(w1)])

  # 每次sess.run 会更新一次w1的滑动平均值
  sess.run(ema_op)
  sess.run([w1,ema.average(w1)])  # average(w1)
  print sess.run([w1,ema.average(w1)])
  #
  sess.run(ema_op)
  sess.run([w1, ema.average(w1)])
  print sess.run([w1,ema.average(w1)])
  # 
  sess.run(ema_op)
  sess.run([w1,ema.average(w1)])
  print sess.run([w1,ema.average(w1)]) # average value of w1 
  
  sess.run(ema_op)
  print sess.run([w1,ema.average(w1)])

  # 
  sess.run(ema_op)
  print sess.run([w1,ema.average(w1)])
  # 
  sess.run(ema_op)
  print sess.run([w1,ema.average(w1)])
  
  # 
  sess.run(ema_op)
  print sess.run([w1,ema.average(w1)])
  sess.run(ema_op)
  print sess.run([w1,ema.average(w1)])

  print ("------------------------------------")
  print ("以上过程实现了滑动平均的实现过程,并且随着迭代次数的增加，权重w逐渐向一开始设定的值逼近!")
  print ("------------------------------------")
  
