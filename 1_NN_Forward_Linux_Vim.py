# -------------  
#coding:utf-8
# Coypright by 
# Modify History: 
#    2018 - 6 -11 
# Purpose:
#     前向传播算法
# 
#  
#------------------------
# 
import tensorflow as tf

#  Just Calulate , But not output 
a  = tf.constant([2.0,9.2])
b  = tf.constant([3.0,4.0])
result = a + b
print result
def int():
	print"%s" %"----**************-------------------------"
        print("a = ", a, " b = ", b)

i =int()
# 下面这个函数是1 ×2 的张量
x = tf.constant([[2.2,2.3]])
# 下面这个w是 2*1 的矩阵
w = tf.constant([[3.4],[2.3]])
print ("x  = \n", x )
print ("w  = \n", w)

# 搭建神经网络但是不计算，会话层做计算
y = tf.matmul(x,w)
print y

# 会话层计算结果
# 
# here use the Seesson to output the result of Matric Multpix 
with tf.Session() as sess:
	print sess.run(y) 
