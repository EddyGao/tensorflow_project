# -*- coding:utf-8 -*-
# load MNIST data
import input_data
import os
import numpy as np
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)
batch_size=50
# start tensorflow interactiveSession
import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def split_pram(w_b_Pram):
    #将数组切分
    temp_arr1 = np.hsplit(w_b_Pram, [200])
    pram_w_1 = temp_arr1[0].reshape([5, 5, 1, 8])
    pram_w_1 = pram_w_1.astype('float32')

    temp_arr2 = np.hsplit(temp_arr1[1], [3200])
    pram_w_2 = temp_arr2[0].reshape([5, 5, 8, 16])
    pram_w_2 = pram_w_2.astype('float32')

    temp_arr3 = np.hsplit(temp_arr2[1], [401408])
    pram_w_f1 = temp_arr3[0].reshape([7 * 7 * 16, 512])
    pram_w_f1 = pram_w_f1.astype('float32')

    temp_arr4 = np.hsplit(temp_arr3[1], [5120])
    pram_w_f2 = temp_arr4[0].reshape([512, 10])
    pram_w_f2 = pram_w_f2.astype('float32')

    return pram_w_1,pram_w_2,pram_w_f1,pram_w_f2
inint_pop=np.load("canshu.npy")
w_conv1_, w_conv2_, w_fc1_, w_fc2_ = split_pram(inint_pop)
print "w_conv1_",w_conv1_
#print "w_conv1_new",w_conv1_new
# convolution
w_conv1_new=tf.convert_to_tensor(w_conv1_)
print w_conv1_new

w_conv2_new=tf.convert_to_tensor(w_conv2_)

w_fc1_new=tf.convert_to_tensor(w_fc1_)

w_fc2_new=tf.convert_to_tensor(w_fc2_)


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Create the model
# placeholder
x = tf.placeholder("float", [batch_size, 784])
y_ = tf.placeholder("float", [batch_size, 10])
# variables
"""
W = tf.Variable(tf.zeros([784,10],"float32"))
b = tf.Variable(tf.zeros([10],"float32"))

y = tf.nn.softmax(tf.matmul(x,W) + b)
"""
# first convolutinal layer
w_conv1 =  tf.Variable(tf.zeros([5, 5, 1, 8],"float32"))
update1=tf.assign(w_conv1,w_conv1_new)
b_conv1 = bias_variable([8])


x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = tf.Variable(tf.zeros([5, 5, 8, 16],"float32"))
update2=tf.assign(w_conv2,w_conv2_new)
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = tf.Variable(tf.zeros([7*7*16,512],"float32"))
update3=tf.assign(w_fc1,w_fc1_new)
b_fc1 = bias_variable([512])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = tf.Variable(tf.zeros([512, 10],"float32"))
update4=tf.assign(w_fc2,w_fc2_new)
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
# train and evaluate the model
#损失函数
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
#cross_entropy = tf.reduce_sum(tf.square(tf.sub(y_,y_conv)))
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-8))

#优化方法
#train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
train_step = tf.train.AdagradOptimizer(1e-3).minimize(cross_entropy)



correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())
#with tf.Session():
    #print w_conv2.eval(sess)

print "init",w_conv1.eval()
sess.run(w_conv1_new)
sess.run(w_conv2_new)
sess.run(w_fc1_new)
sess.run(w_fc2_new)
sess.run(update1)
sess.run(update2)
sess.run(update3)
sess.run(update4)
print "update",  sess.run(w_conv1)

for i in range(200000):
    batch = mnist.train.next_batch(batch_size)

    #print "the %d" % (i)
    #print w_conv1.eval()
    if i%100 == 0:

        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print "step %d, train accuracy %g" %(i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print "test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
