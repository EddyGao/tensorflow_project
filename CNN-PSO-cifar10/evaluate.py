# -*- coding:utf-8 -*-

from sys import path
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
path.append('../..')
import extract_cifar10
import inspect_image

result=np.zeros(500)
# 初始化单个卷积核上的参数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化单个卷积核上的偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积操作
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def split_pram(w_Pram):
    #将数组切分
	temp_arr1= np.hsplit(w_Pram, [4800])
	pram_w_1 = temp_arr1[0].reshape([5,5,3,64])
	pram_w_1 = pram_w_1.astype('float32')

	temp_arr2 = np.hsplit(temp_arr1[1],[102400])
	pram_w_2 = temp_arr2[0].reshape([5,5,64,64])
	pram_w_2 = pram_w_2.astype('float32')

	temp_arr3 = np.hsplit(temp_arr2[1] , [1572864])
	pram_w_f1 = temp_arr3[0].reshape([8*8*64,384])
	pram_w_f1 = pram_w_f1.astype('float32')

	temp_arr4 = np.hsplit(temp_arr3[1] , [73728])
	pram_w_f2 = temp_arr4[0].reshape([384,192])
	pram_w_f2 = pram_w_f2.astype('float32')

	temp_arr5 = np.hsplit(temp_arr4[1] , [1920])
	pram_W_f2 = temp_arr5[0].reshape([192, 10])
	pram_W_f2 = pram_W_f2.astype('float32')

	return pram_w_1, pram_w_2, pram_w_f1, pram_w_f2, pram_W_f2

inint_pop=np.load("cifar10_canshu.npy")
w_conv1_, w_conv2_,w_fc1_, w_fc2_ ,W_out_= split_pram(inint_pop)


#print "w_conv1_new",w_conv1_new
# convolution
w_conv1_new=tf.convert_to_tensor(w_conv1_)
w_conv2_new=tf.convert_to_tensor(w_conv2_)
w_fc1_new=tf.convert_to_tensor(w_fc1_)
w_fc2_new=tf.convert_to_tensor(w_fc2_)
W_out_new=tf.convert_to_tensor(W_out_)

sess = tf.InteractiveSession()

# 声明输入图片数据，类别
x = tf.placeholder('float', [None, 32, 32, 3])
y_ = tf.placeholder('float', [None, 10])

# 第一层卷积层
W_conv1 = weight_variable([5, 5, 3, 64])
update1=tf.assign(W_conv1,w_conv1_new)
b_conv1 = bias_variable([64])
# 进行卷积操作，并添加relu激活函数
conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
# pool1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
# norm1
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

# 第二层卷积层
W_conv2 = weight_variable([5, 5, 64, 64])
update2=tf.assign(W_conv2,w_conv2_new)
b_conv2 = bias_variable([64])
conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)
# norm2
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
# pool2
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

# 全连接层
# 权值参数
W_fc1 = weight_variable([8 * 8 * 64, 384])
update3=tf.assign(W_fc1,w_fc1_new)
# 偏置值
b_fc1 = bias_variable([384])
# 将卷积的产出展开
pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
# 神经网络计算，并添加relu激活函数
fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1)

# 全连接第二层
# 权值参数
W_fc2 = weight_variable([384, 192])
update4=tf.assign(W_fc2,w_fc2_new)
# 偏置值
b_fc2 = bias_variable([192])
# 神经网络计算，并添加relu激活函数
fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)

# Dropout层，可控制是否有一定几率的神经元失效，防止过拟合，训练时使用，测试时不使用
keep_prob = tf.placeholder("float")
# Dropout计算
fc1_drop = tf.nn.dropout(fc2, keep_prob)

# 输出层，使用softmax进行多分类
W_out = weight_variable([192, 10])
update5=tf.assign(W_out ,W_out_new)
b_out = bias_variable([10])
y_conv = tf.maximum(tf.nn.softmax(tf.matmul(fc1_drop, W_out) + b_out), 1e-30)

# 补丁，防止y等于0，造成log(y)计算出-inf
# y1 = tf.maximum(y_conv,1e-30)

# 代价函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 使用Adam优化算法来调整参数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 测试正确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 保存模型训练数据
saver = tf.train.Saver()

# 所有变量进行初始化
sess.run(tf.initialize_all_variables())
# / home / uu / xueqian / new_cifar10 / cifar - 10 - batches - bin
# 获取cifar10数据
# cifar10_data_set = extract_cifar10.Cifar10DataSet('../../data/')
cifar10_data_set = extract_cifar10.Cifar10DataSet('/home/uu/xueqian/')
test_images, test_labels = cifar10_data_set.test_data()

# 进行训练
start_time = time.time()

sess.run(w_conv1_new)
sess.run(w_conv2_new)
sess.run(w_fc1_new)
sess.run(w_fc2_new)
sess.run(W_out_new)
sess.run(update1)
sess.run(update2)
sess.run(update3)
sess.run(update4)
sess.run(update5)
for i in xrange(10000):
    # 获取训练数据
    # print i,'1'
    batch_xs, batch_ys = cifar10_data_set.next_train_batch(50)
    # print i,'2'

    # 每迭代100个 batch，对当前训练数据进行测试，输出当前预测准确率
    if i % 100 == 0:
        print time.strftime("%Y-%m-%d %X", time.localtime())
        # print "test accuracy %g"%accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)
        result[i / 100]=train_accuracy
        # 计算间隔时间
        end_time = time.time()
        print 'time: ', (end_time - start_time)
        start_time = end_time

    if (i + 1) % 10000 == 0:
        # 输出整体测试数据的情况
        avg = 0
        for j in xrange(20):
            avg += accuracy.eval(
                feed_dict={x: test_images[j * 50:j * 50 + 50], y_: test_labels[j * 50:j * 50 + 50], keep_prob: 1.0})
        avg /= 20
        print "test accuracy %g" % avg
        # 保存模型参数
        if not tf.gfile.Exists('model_data'):
            tf.gfile.MakeDirs('model_data')
        save_path = saver.save(sess, "model_data/model.ckpt")
        print "Model saved in file: ", save_path

    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
np.save("cifar10_result1.npy",result)
plt.plot(result)
plt.show()

# 输出整体测试数据的情况
avg = 0
for i in xrange(200):
    avg += accuracy.eval(
        feed_dict={x: test_images[i * 50:i * 50 + 50], y_: test_labels[i * 50:i * 50 + 50], keep_prob: 1.0})
avg /= 200
print "test accuracy %g" % avg



