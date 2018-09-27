#coding:utf-8
# 
#利用模块化思想搭建神经网络.
#     执行先后顺序：
#           1.opt4_8_generateds.py  - 准备数据集
#           2.opt4_8_forward.py     - 前向传播算法搭建神经网络并通过sess.run 计算结果
#           3.opt4_8_backward.py    - 后先传播算法喂入大量数据到NN,更新参数值使得代价函数最小化
# 
#搭建神经网络的模块1： 
#       opt4_8_generateds.py
# --------------------------------------------
# 
# --------------------------------------------
# 导入模块生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
seed  = 2 
def generateds():
   # 基于seed 产生随机数
   rdm  = np.random.RandomState(seed)
   X    = rdm.randn(300,2)  # (x0,x1) 作为输入数据集,300行数2列
   # x0*x0 + x1*x1 < 2时，将Y_ 赋值为1，>2 时将Y_ 赋值为0 
   Y_  = [int( x0*x0 + x1*x1 < 2 ) for (x0, x1 ) in X]
  
   # 1 赋值成'red', 0赋值为'blue' 别于可视化区分
   Y_c =[ ['red' if y else 'blue'] for y in Y_ ]
   
   # 对数据集X和标签Y进行形状整理，第一个元素为-1 表示行数不受限制；X 为2列，Y 为1列
   X  = np.vstack(X).reshape(-1,2)
   Y_ = np.vstack(Y_).reshape(-1,1)
  
   return X,Y_,Y_c

