# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import input_data
import tensorflow as tf
import gc
import os
import time
from time import strftime
# load MNIST data
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)
# 参数必须得来自pso的返回值才行，按照这样的形式更改代码，应该
sess = tf.InteractiveSession()
#每个粒子的参数总数
x_num = 409928
# 分别是粒子的个体和社会的学习因子，也称为加速常数
lr = (1.80000,2.0000000)
max_step=30
# 种群规模
sizepop =80
# 粒子的位置的范围限制,x、y方向的限制相同
rangepop = (-2,2)
# 粒子的速度范围限制
rangespeed = (-1,1)

w=0.5
def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def split_pram(w_b_Pram):
    #将数组切分
    temp_arr1= np.hsplit(w_b_Pram, [200])
    pram_w_1 = temp_arr1[0].reshape([5,5,1,8])
    pram_w_1 = pram_w_1.astype('float32')


    temp_arr2 = np.hsplit(temp_arr1[1],[3200])
    pram_w_2 = temp_arr2[0].reshape([5,5,8,16])
    pram_w_2 = pram_w_2.astype('float32')


    temp_arr3 = np.hsplit(temp_arr2[1] , [401408])
    pram_w_f1 = temp_arr3[0].reshape([7*7*16,512])
    pram_w_f1 = pram_w_f1.astype('float32')


    temp_arr4 = np.hsplit(temp_arr3[1] , [5120])
    pram_w_f2 = temp_arr4[0].reshape([512,10])
    pram_w_f2 = pram_w_f2.astype('float32')



    return pram_w_1,pram_w_2,pram_w_f1,pram_w_f2

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_ = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
# variables

# Create the model,隐式的表达式
# first convolutinal layer
w_conv1 = tf.placeholder("float", [5, 5, 1, 8])
b_conv1 = bias_variable([8])

x_image = tf.reshape(x_, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = tf.placeholder("float", [5, 5, 8, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = tf.placeholder("float", [7 * 7 * 16, 512])
b_fc1 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# readout layer
w_fc2 = tf.placeholder("float", [512, 10])
b_fc2 =bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-10))
#cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = tf.reduce_sum(tf.square(tf.sub(y_,y_conv)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())
def func(x):
    batch = mnist.train.next_batch(50)

    # x输入粒子位置
    w_conv1_new, w_conv2_new,w_fc1_new,  w_fc2_new  = split_pram(x)
    #适应度函数的常量输入
    #print  sess.run(cross_entropy,
                 #feed_dict={x_: batch[0], y_: batch[1], keep_prob: 1.0, w_conv1: w_conv1_new, w_conv2: w_conv2_new,
                            #w_fc1: w_fc1_new, w_fc2: w_fc2_new})

    a= sess.run(accuracy,feed_dict={x_:batch[0],y_:batch[1],keep_prob:1.0,w_conv1:w_conv1_new,w_conv2:w_conv2_new,
                                      w_fc1:w_fc1_new,w_fc2:w_fc2_new})

    return a

def initpopvfit(sizepop):
    pop = np.ones((sizepop,x_num)) #x_num个参数

    print type(pop)
    v = np.ones((sizepop,x_num))
    fitness = np.zeros(sizepop)

    for i in xrange(sizepop):
        for k in xrange(x_num):
            pop[i][k] = np.random.rand()*(rangepop[1]-rangepop[0])+rangepop[0]
            v[i][k] = np.random.rand()*(rangespeed[1]-rangespeed[0])+rangespeed[0]
        #fitness[i] = func(pop[i])
        #pop[i]=np.random.lognormal(0.0,0.1,[x_num])
        fitness[i] = func(pop[i])
    return pop,v,fitness

def getinitbest(fitness,pop):
    # 群体最优的粒子位置及其适应度值
    pbestpop_,pbestfitness_ = pop,fitness
    gbestpop_,gbestfitness_ = pop[fitness.argmax()],fitness.max()
    #个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似


    return gbestpop_,gbestfitness_,pbestpop_,pbestfitness_

pop, v, fitness = initpopvfit(sizepop)
result = np.zeros(max_step)
gbestpop, gbestfitness, pbestpop, pbestfitness = getinitbest(fitness, pop)

def optimize(h,gbestpop,pop, gbestfitness, pbestpop, pbestfitness,v):

    w=0.9-0.03*h
    # 速度更新
    for j in xrange(0,sizepop):
        v[j]=w*v[j]+lr[0]* np.random.rand() * (pbestpop[j] - pop[j])+ lr[1]* np.random.rand() * (gbestpop - pop[j])
    #v[v < rangespeed[0]] = rangespeed[0]
    #v[v > rangespeed[1]] = rangespeed[1]
    # 粒子位置更新
    print ("粒子位置更新")
    for j in xrange(0,sizepop):
        #pop[j] += 0.5 * v[j]
        pop[j] = 0.7*(0.5*v[j])+0.3*pop[j]
        fitness[j] = func(pop[j])
        if fitness[j] > pbestfitness[j]:
            pbestfitness[j] = fitness[j]
            pbestpop[j] = pop[j]
    #pop[pop < rangepop[0]] = rangepop[0]
    #pop[pop > rangepop[1]] = rangepop[1]
    if pbestfitness.max() > gbestfitness:
        gbestfitness = pbestfitness.max()
        gbestpop = pop[pbestfitness.argmax()]
        # gbestpop = pop[pbestfitness.argmax()]

    return gbestpop,pop, gbestfitness, pbestpop, pbestfitness,v

for i in xrange(0,max_step):

    gbestpop, pop, gbestfitness, pbestpop, pbestfitness,v=optimize(i,gbestpop,pop, gbestfitness, pbestpop, pbestfitness,v)
    result[i] = gbestfitness

    print 'epoch->' ,i ,'cross_entropy->',gbestfitness
    time.sleep(1)
    gc.collect()

np.save("canshu.npy",gbestpop)
print gbestpop

plt.plot(result)
plt.show()
