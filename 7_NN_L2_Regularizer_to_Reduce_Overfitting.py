#coding:utf-8
# ----------------------------------------
# 以下将实现正则化对过拟合的缓解过程
# -----------------------------------------
#
# 正则化在损失函数中引入了了模型复杂度的指标，利用给w加权重来弱化训练数据的噪声（一般不正则化参数b）
# loss = loss (y 与 y_) + Regularizer * loss (w)
# loss(y 与 y_) 表示模型中所有参数的损失函数，，如MMSE/交叉熵 , 超参数regularizer 是用来给出参数w在总的Loss中所占有的比例，也就是正则化的权重
# loss(w) 表示需要正则化的参数

# loss(w) = tf.contrib.layers.l1_regularizer(REGULARIZER)(w)  --- loss(w1) = sum(|wi|)
# loss(w) = tf.contrib.layers.l2_regularizer(REGULARIZER)(w)  --- loss(w2) = sum(|wi|^2)

# 把对w权重正则化以后的数据加到losses集合中
# tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))

# 并对总的loss求和
# loss  = cem(交叉熵) + tf.add_n(tf.get_collection('losses'))

# -----------------------------------------
# 实现背景：
# 随机产生300 个X[x0,x1]的正态随机分布点：
# 标注Y_ 当x0^2 + x1^2 < 2 时y_ = 1 （红）, 其余y_ = 0 (蓝)
# import matplotlib.pyplot as plt      -  sudo pip install + 模块
# plt.scatter(x坐标，y坐标，c = '颜色')
# plt.show()
#
# 对x轴和y轴的开始和结束点，以及中间的步长打点形成网格区域 
# xx, yy  = np.mgrid[start:end:steps, start:end:steps]

# 将x轴和y轴拉直, x轴形成一维，y轴形成一维矩阵，并把x 与y 轴一一配对形成网格坐标点，并把这些网格坐标点喂入神经网络
# grid  = np.c_[xx.ravel(), yy.ravel()] 

# 将上述生成的网格坐标点喂入神经网络，生成的probs就是区域中所有偏红还是偏蓝的量化值
# probs  = sess.run(y,feed_dict =  {x: grid})
# 将probs 整形成跟xx 相同的维数
# probs  = probs.reshape(xx.shape)

# plt.contour(x轴坐标，y轴坐标，该点的高度，levels = [等高线的高度]) # 用levels将指定高对的点描绘上颜色
# plt.show() # 将点都画出来
# 画出数据的决策边界 - x1**2 + x2**2 <= r**2

#-------------------------------------------
# 导入模块，生成模拟数据集
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
BATCH_SIZE  = 300 
seed        = 2 

# 基于seed 生成随机数据
rdm  = np.random.RandomState(seed)
# 随机返回(300,2)的随机数作为输入数据（x0,x1）
X    = rdm.randn(300,2)
# 标记正确答案，认为每一行中x0^2+ x1^2 <2 的为红色，标记Y_ = 1,在圆外面的标记为 Y_  = 0; 
Y_   = [int (x0*x0 + x1*x1 < 2) for (x0,x1) in X]
print 'Y_  = \n ',Y_ 

# 对应Y_,生成数据Y_c,将1赋值成'red', 0赋值成'blue'; 这样可视化显示时，人可以直观区分
Y_c  =  [ ['red' if y else 'blue'] for y in Y_]

# 将X和Y_ 重新整理, X重新整理为n行2列。Y_重新整理成n行1列
X    = np.vstack(X).reshape(-1,2)   # N*2 
Y_   = np.vstack(Y_).reshape(-1,1)  # N*1
Y_c  = np.vstack(Y_c).reshape(-1,1) # N*1

#----------------
print '随机返回(300,2)的随机数作为输入数据（x0,x1) , X =  :\n', X
print '标记正确答案，认为每一行中x0^2+ x1^2 <2 的为红色，标记Y_ = 1,在圆外面的标记为 Y_  = :\n', Y_
print '对应Y_,生成数据Y_c,将1赋值成red, 0赋值d成blue, Y_c = :\n', Y_c

# ----------------
# 用plt.scatter画出数据集中第0列和第1列元素的点即各行的(x0,x1) ，用各行的Y_c对应的值来表示颜色(c = color颜色)
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c)) # x0 表示横坐标；x1表示纵坐标, c= 颜色 ; Y_ 中的1 表示红色， 
plt.show()

# 定义神经网络的输入，参数和输出，定义前向传播过过程
def get_weight(shape,regularizer):
  w  = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
  tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) # 正则化参数w
  return w 

def get_bias(shape):
  b  = tf.Variable(tf.constant(0.01,shape  = shape,dtype = tf.float32 ))
  return b 

# 为输入数据x和标签y_输入值占位
x  = tf.placeholder(tf.float32, shape = (None,2 ))
y_ = tf.placeholder(tf.float32, shape = (None,1 ))

# 生成权重值w1 = [2,11 ], 正则化参数regularizer  = 0.01 
w1 = get_weight([2,11], 0.01)           # 神经网络为2个输入x1,x2 隐藏层有11个元素
b1 = get_bias([11])                     # 偏执单元值是一个常数，个数是11个。等于隐藏层的个数11
y1 = tf.nn.relu(tf.matmul(x,w1) + b1 )  # 11个元素对应相加

# 第二层神经网络，隐藏层有11个元素，直接输出一个y_out ,所以这里w2是11×1 的矩阵
w2 = get_weight([11,1], 0.01)
b2 = get_bias([1]) # 第二层的偏执单元个数是1个,tf.constant, dtype = float32 

# 这一层直接输出，输出层不过激活函数ReLu 
y  = tf.matmul(y1,w2) + b2 
# -------------

# 定义代价函数
loss_mse    = tf.reduce_mean(tf.square(y - y_))  # mmse 算法定义代价函数
#
# 这里的total_loss 表示引入对w正则化后的全部loss error
loss_total  = loss_mse  + tf.add_n(tf.get_collection('losses'))

# 定义反向传播算法-  不含正则化, 否则minimize(loss_total)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)


# 训练过程
with tf.Session() as sess:
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  STEPS  = 40000
  for i in range(STEPS):
    start  = (i *BATCH_SIZE)* 300
    end    = start + BATCH_SIZE 
    sess.run(train_step, feed_dict = {x:X[start:end],  y_:Y_[start:end] })
    if i % 2000 == 0: 
      loss_mse_value  = sess.run(loss_mse, feed_dict = {x:X, y_:Y_} )
      print("After %d train steps , current loss is %f" %(i, loss_mse_value))
  ## xx 在-3~+3 步长 = 0.01 , yy 在-3~+3 步长 = 0.01 ，生成二维网格坐标点
  xx,yy = np.mgrid[-3:3:0.01, -3:3:0.01]

  # 将xx,yy 拉直，并合并成一个2列的矩阵，得到一个网络坐标点的结合
  grid  = np.c_[xx.ravel(), yy.ravel()]

  # 将坐标点喂入神经网络NN，probs 是输出, y 是前向传播算法的输出值y 
  probs = sess.run(y,feed_dict = {x:grid})
  # 将probs 调整xx的样子
  probs = probs.reshape(xx.shape)
  
  # print -----
  print('没有正则化的参数:')
  print "w1: \n",sess.run(w1)
  print "b1: \n",sess.run(b1)
  print "w2: \n",sess.run(w2)
  print "b2: \n",sess.run(b2)

  # ----------
plt.scatter(X[:,0], X[:,1],c=np.squeeze(Y_c) )
plt.contour(xx,yy,probs,levels = [0.5])
plt.show()


# 定义反向传播函数，包含正则化 - loss_total
train_step_1  = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  STEPS   = 40000
  for i in range(STEPS):
    start = (i*BATCH_SIZE)%300 
    end   = start + BATCH_SIZE
    sess.run(train_step_1, feed_dict = {x:X[start:end],   y_:Y_[start:end]} )
    if i%2000 == 0:
      loss_mse_regularizer_value   = sess.run(loss_total,feed_dict= {x:X,y_:Y_})
      print("After %d train steps, loss with regularizer is %f " %(i, loss_mse_regularizer_value))

  xx,yy  = np.mgrid[-3:3:0.01, -3:3:0.01]
  grid   = np.c_[xx.ravel(), yy.ravel()]
  probs  = sess.run(y,feed_dict = {x:grid})
  probs  = probs.reshape(xx.shape)

#--------------------
#
  print ('使用正则化后的训练过程')
  print "w1: \n",sess.run(w1)
  print "b1: \n",sess.run(b1)
  print "w2: \n",sess.run(w2)
  print "b2: \n",sess.run(b2)
# end of session 

plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))  # 
plt.contour(xx,yy,probs,levels = [0.5]) # 对probs = 0.5 的所有点上色
plt.show()
































