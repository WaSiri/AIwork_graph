import tensorflow as tf
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimage
import cv2 as cv
import time
import scipy.io as scio
import pandas as pd

# 数据切分函数
class DataSet(object):

    def __init__(self, images, labels, num_examples):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0  # 完成遍历轮数
        self._index_in_epochs = 0  # 调用next_batch()函数后记住上一次位置
        self._num_examples = num_examples  # 训练样本数

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epochs

        if self._epochs_completed == 0 and start == 0 and shuffle:
            index0 = np.arange(self._num_examples)
            np.random.shuffle(index0)
            self._images = np.array(self._images)[index0]
            self._labels = np.array(self._labels)[index0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            if shuffle:
                index = np.arange(self._num_examples)
                np.random.shuffle(index)
                self._images = self._images[index]
                self._labels = self._labels[index]
            start = 0
            self._index_in_epochs = batch_size - rest_num_examples
            end = self._index_in_epochs
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)

        else:
            self._index_in_epochs += batch_size
            end = self._index_in_epochs
            return self._images[start:end], self._labels[start:end]

#读取数据
data_path="./face_train.mat"
data_train = scio.loadmat(data_path)

# 获得训练数据并进行数据处理
origin_data_train_data = data_train.get('data')
# print(origin_data_train_data.shape)
data_train_data = np.reshape(origin_data_train_data,(-1,48,48,1))
# print(data_train_data.shape)
data_train_data_resize = []
for i in range(data_train_data.shape[0]):
    # if i % 1000 == 0:
    #     print('now resize 48 --> 256 train set', i)
    data_train_data_resize.append(cv.resize(data_train_data[i], (256, 256), interpolation=cv.INTER_LINEAR))
data_train_data_end = np.array(data_train_data_resize)
# print(data_train_data_end.shape)
data_train_data = np.reshape(data_train_data_end,(-1,65536))

# 获取训练数据标签并处理
data_train_label = data_train.get('label')
# print(data_train_label.shape)

# 将已获得的数据传给切分函数
ds = DataSet(data_train_data,data_train_label,28709)

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
#  输入大小为48*48
x = tf.placeholder(tf.float32, shape=[None,65536])
#  初始化输出Y
#  [0,6]共七个分类
y_ = tf.placeholder(tf.float32,shape=[None,7])

W_conv1_1 =  weight_variable([3,3,1,32])
b_conv1_1 = bias_variable([32])
x_image = tf.reshape(x,[-1,256,256,1])
# relu激化和池化
h_conv1_1 = tf.nn.relu(conv2d(x_image,W_conv1_1) + b_conv1_1)

W_conv1_2 = weight_variable([3,3,32,64])
b_conv1_2 = bias_variable([64])
h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1,W_conv1_2) + b_conv1_2)
h_pool1 = max_pool_2x2(h_conv1_2)

W_conv2_1 = weight_variable([3,3,64,128])
b_conv2_1 = bias_variable([128])
h_conv2_1 = tf.nn.relu(conv2d(h_pool1,W_conv2_1) + b_conv2_1)

W_conv2_2 = weight_variable([3,3,128,128])
b_conv2_2 = bias_variable([128])
h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1,W_conv2_2) + b_conv2_2)
h_pool2 = max_pool_2x2(h_conv2_2)

W_conv3_1 = weight_variable([3,3,128,256])
b_conv3_1 = bias_variable([256])
h_conv3_1 = tf.nn.relu(conv2d(h_pool2,W_conv3_1) + b_conv3_1)

W_conv3_2 = weight_variable([3,3,256,256])
b_conv3_2 = bias_variable([256])
h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1,W_conv3_2) + b_conv3_2)

W_conv3_3 = weight_variable([3,3,256,256])
b_conv3_3 = bias_variable([256])
h_conv3_3 = tf.nn.relu(conv2d(h_conv3_2,W_conv3_3) + b_conv3_3)

W_conv3_4 = weight_variable([3,3,256,256])
b_conv3_4 = bias_variable([256])
h_conv3_4 = tf.nn.relu(conv2d(h_conv3_3,W_conv3_4) + b_conv3_4)
h_pool3 = max_pool_2x2(h_conv3_4)

W_conv4_1 = weight_variable([3,3,256,512])
b_conv4_1 = bias_variable([512])
h_conv4_1 = tf.nn.relu(conv2d(h_pool3,W_conv4_1) + b_conv4_1)

W_conv4_2 = weight_variable([3,3,512,512])
b_conv4_2 = bias_variable([512])
h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1,W_conv4_2) + b_conv4_2)

W_conv4_3 = weight_variable([3,3,512,512])
b_conv4_3 = bias_variable([512])
h_conv4_3 = tf.nn.relu(conv2d(h_conv4_2,W_conv4_3) + b_conv4_3)

W_conv4_4 = weight_variable([3,3,512,512])
b_conv4_4 = bias_variable([512])
h_conv4_4 = tf.nn.relu(conv2d(h_conv4_3,W_conv4_4) + b_conv4_4)
h_pool4 = max_pool_2x2(h_conv4_4)

W_conv5_1 = weight_variable([3,3,512,512])
b_conv5_1 = bias_variable([512])
h_conv5_1 = tf.nn.relu(conv2d(h_pool4,W_conv5_1) + b_conv5_1)

W_conv5_2 = weight_variable([3,3,512,512])
b_conv5_2 = bias_variable([512])
h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1,W_conv5_2) + b_conv5_2)

W_conv5_3 = weight_variable([3,3,512,512])
b_conv5_3 = bias_variable([512])
h_conv5_3 = tf.nn.relu(conv2d(h_conv5_2,W_conv5_3) + b_conv5_3)

W_conv5_4 = weight_variable([3,3,512,512])
b_conv5_4 = bias_variable([512])
h_conv5_4 = tf.nn.relu(conv2d(h_conv5_3,W_conv5_4) + b_conv5_4)
h_pool5 = max_pool_2x2(h_conv5_4)

# 全连接层
# 该层拥有4608个神经元（神经元个数可在0~4000间调参）
# W的第1维size为6*6*128，6*6是h_pool2输出的size，64是第2层输出神经元个数
w_fc1 = weight_variable([8 * 8 * 512, 4608])
b_fc1 = bias_variable([4608])
h_pool5_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat,w_fc1) + b_fc1)

# Dropout减少过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

w_fc2 = weight_variable([4608, 512])
b_fc2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)

# 读出层
W_fc3 = weight_variable([512,7])
b_fc3 = bias_variable([7])
y_conv = tf.matmul(h_fc2_drop,W_fc3) + b_fc3

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y_conv)
    )

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saver = tf.train.Saver()

# 运行
# start = time.process_time()
# ydrop = []
# averagedroplist = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batch = ds.next_batch(50)
        if i % 100 == 0:
            # average_accuracy = np.mean(averagedroplist)
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            # ydrop.append(average_accuracy)
            # averagedroplist = []
        # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        # averagedroplist.append(train_accuracy)
        # 更新学习率
        train_step.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0 })

# elapsed = (time.process_time() - start)
# print("Time used:", elapsed)
#
#
# start = time.process_time()
# ynodrop = []
# averagenodroplist = []
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(2000):
#         batch = mnist.train.next_batch(50)
#         if i % 20 == 0:
#             average_accuracy = np.mean(averagenodroplist)
#             print('step %d, training accuracy %g' % (i, average_accuracy))
#             ynodrop.append(average_accuracy)
#             averagenodroplist = []
#         train_accuracy = BP_accuarcy(y_conv_nodrop).eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
#         averagenodroplist.append(train_accuracy)
#         # 更新学习率
#         train_step_nodrop.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5 })
#
#     # 保存模型
#     # saver.save(sess, 'D:/CSU/program/Python/Practice/Vs2019/my_mnist/my_mnist/model.ckpt')
#
# elapsed = (time.process_time() - start)
# print("Time used:", elapsed)
# xrange = list(range(100))
# plt.title('Result Analysis')
# plt.plot(xrange, ydrop, "-.", color='green', label='usedrop')
# plt.plot(xrange, ynodrop, "-.", color='red', label='nonedrop')
# plt.legend()
# plt.xlabel('times')
# plt.ylabel('fitness')
# plt.show()






