import tensorflow as tf
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import time


#  导入mnist数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print(mnist.train.labels.shape)


#  CNN权重
#  用对称破坏的小噪声初始化权重
def weight_variable(shape):
    initial =  tf.random.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#  偏置参数
def bias_variable(shape):  # 将偏置参数初始化为小的正数，以避免死神经元
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

#  卷积核
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#  池化层（采用最大池化）
def max_pool_2x2(x):
    return tf.nn.max_pool2d(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#  初始化输入X
#  输入大小为28*28
x = tf.placeholder(tf.float32, shape=[None,784])
#  初始化输出Y
#  因为MNIST为[0,9]共十个分类
y_ = tf.placeholder(tf.float32,shape=[None,10])

# 第一个卷积层
# 创建卷积核W_conv1,表示卷积核大小为5*5，第一层网络的输入和输出神经元个数分别为1和32
W_conv1 =  weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])
# relu激化和池化
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
# h_pool1的输出即为第一层网络输出，shape为[batch,14,14,32]
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷积层
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
# 该层拥有1024个神经元（神经元个数可在0~4000间调参）
# W的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#  将第2层的输出reshape成[batch, 7*7*64]的张量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

# Dropout减少过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


# 读出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y_conv)
    )
mean_square_error= tf.reduce_mean(
    tf.compat.v1.losses.mean_squared_error(y_,y_conv)
    )
train_step_CE = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step_MSE = tf.train.AdamOptimizer(1e-4).minimize(mean_square_error)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 反向算法更新率
def BP_loss(y_predict):
    # 用交叉熵作为代价函数
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_predict)
    )
    return cross_entropy

# 评估函数
# 比直接调用eval，大概慢5到1倍
def BP_accuarcy(y_predict):
    # 评估函数
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_, 1))

    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saver = tf.train.Saver()

# 运行
start = time.process_time()
yCE = []
averageCElist = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 20 == 0:
            average_accuracy = np.mean(averageCElist)
            print('step %d, training accuracy %g' % (i, average_accuracy))
            yCE.append(average_accuracy)
            averageCElist = []
        train_accuracy = BP_accuarcy(y_conv).eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        averageCElist.append(train_accuracy)
        # 更新学习率
        train_step_CE.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5 })

    # saver.save(sess, './relu_drop/model.ckpt')

elapsed = (time.process_time() - start)
print("use CE time used:", elapsed)


start = time.process_time()
yMSE = []
averageMSElist = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 20 == 0:
            average_accuracy = np.mean(averageMSElist)
            print('step %d, training accuracy %g' % (i, average_accuracy))
            yMSE.append(average_accuracy)
            averageMSElist = []
        train_accuracy = BP_accuarcy(y_conv).eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        averageMSElist.append(train_accuracy)
        # 更新学习率
        train_step_MSE.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5 })

    # 保存模型
    # saver.save(sess, './relu_nodrop/model.ckpt')

elapsed = (time.process_time() - start)
print("use MSE Time used:", elapsed)
xrange = list(range(100))
plt.title('Result Analysis')
plt.plot(xrange, yCE, "-.", color='green', label='CE')
plt.plot(xrange, yMSE, "-.", color='red', label='MSE')
plt.legend()
plt.xlabel('times')
plt.ylabel('fitness')
plt.show()


