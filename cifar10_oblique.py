import tensorflow as tf
import batch_norm
import gutils

INPUT_NODE=3072#输入节点数，就是图像的大小
OUTPUT_NODE=10#输出节点数，就是多分类问题
IMAGE_SIZE=32
NUM_CHANNELS=3
NUM_LABELS=10
#第一卷积层的尺寸
CONV1_DEEP=64#卷积层深度
CONV1_SIZE=5#卷积层尺寸
#第二层卷积的尺寸
CONV2_DEEP=64
CONV2_SIZE=5
#全连接层节点个数
FC_SIZE=512
#定义训练的前向传播，其中加入了dropout过程

def inference(input_tensor,train,regularizer):
    input_tensor = batch_norm.batch_norm(input_tensor)

    with tf.variable_scope('layer1-conv1_oblique'):
        conv1_weights_o = tf.get_variable("weight_o", shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.assign(conv1_weights_o , gutils.unit(conv1_weights_o))
        conv1_biases_o=tf.get_variable("biases_o",shape=[CONV1_DEEP],initializer=tf.constant_initializer(0.0))

        conv1_weights_o_tmp=tf.get_variable("weight_o_tmp",shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=1))
        conv1_biases_o_tmp = tf.get_variable("biases_o_tmp", shape=[CONV1_DEEP],
                                           initializer=tf.constant_initializer(0.0))

    #卷积网络前向传播，这里步长为1且做全0填充，输出是28*28*32的矩阵，步幅就在第二个参数矩阵里面了。
        conv1_o = tf.nn.conv2d(input_tensor, conv1_weights_o, strides=[1, 1, 1, 1], padding='SAME')
        conv1_batch_o = batch_norm.batch_norm(conv1_o, scale=None)
        relu1_o_oblique = tf.nn.relu(tf.nn.bias_add(conv1_batch_o, conv1_biases_o))

    with tf.name_scope('layer2-pool1_oblique'):
        pool1_o = tf.nn.max_pool(relu1_o_oblique, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 第二个参数是步幅，第三个参数是步长
        pool1_batch_o_oblique = batch_norm.batch_norm(pool1_o, scale=None)

    with tf.variable_scope('layer3-conv2_oblique'):
        conv2_weights_o = tf.get_variable('weight_o', shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.assign(conv2_weights_o , gutils.unit(conv2_weights_o))
        conv2_biases_o = tf.get_variable('biases_o', shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2_weights_o_tmp = tf.get_variable("weight_o_tmp", shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                              initializer=tf.truncated_normal_initializer(stddev=1))

        conv2_biases_o_tmp = tf.get_variable("biases_o_tmp", shape=[CONV2_DEEP],
                                           initializer=tf.constant_initializer(0.0))

    #卷积网络前向传播
        conv2_o = tf.nn.conv2d(pool1_batch_o_oblique, conv2_weights_o, strides=[1, 1, 1, 1], padding='SAME')
        conv2_batch_o = batch_norm.batch_norm(conv2_o, scale=None)
        relu2_o_oblique = tf.nn.relu(tf.nn.bias_add(conv2_batch_o, conv2_biases_o))

    with tf.name_scope('layer4-pool2_oblique'):
        pool2_o = tf.nn.max_pool(relu2_o_oblique, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 第二个参数是步幅，第三个参数是步长
        pool2_batch_o_oblique = batch_norm.batch_norm(pool2_o, scale=None)

    with tf.variable_scope('layer5-fc1_oblique'):
        pool_shape = pool2_batch_o_oblique.get_shape().as_list()
        # pool_shape的第一个数据pool_shape[0]就是batch的大小
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        # 重新改变输入的结构把它拉成一个向量做全连接
        reshaped_o = tf.reshape(pool2_batch_o_oblique, [-1, nodes])

        fc1_weights_o = tf.get_variable('weight_o', shape=[nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        tf.assign(fc1_weights_o , gutils.unit(fc1_weights_o))

        fc1_weights_o_tmp = tf.get_variable('weight_o_tmp',
                                            shape=[nodes, FC_SIZE],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))

    #只对全连接参数做正则化
        if regularizer!=None:
            tf.add_to_collection('losses_o_oblique', regularizer(fc1_weights_o))
        fc1_biases_o = tf.get_variable('biases_o', shape=[FC_SIZE], initializer=tf.constant_initializer(0.0))
        fc1_biases_o_tmp = tf.get_variable("biases_o_tmp", shape=[FC_SIZE],
                                           initializer=tf.constant_initializer(0.0))

        fc1_o_oblique = tf.nn.relu(tf.matmul(reshaped_o, fc1_weights_o) + fc1_biases_o)

        if train:
            fc1_o_oblique = tf.nn.dropout(fc1_o_oblique, 0.5)

    with tf.variable_scope('layer6-fc2_oblique'):
        fc2_weights_o = tf.get_variable('weight_o', shape=[FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.assign(fc2_weights_o , gutils.unit(fc2_weights_o))

        fc2_weights_o_tmp = tf.get_variable('weight_o_tmp',
                                            shape=[FC_SIZE,NUM_LABELS],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer!=None:
            tf.add_to_collection('losses_o_oblique', regularizer(fc2_weights_o))

        fc2_biases_o = tf.get_variable('biases_o', shape=[NUM_LABELS], initializer=tf.constant_initializer(0.0))
        fc2_biases_o_tmp = tf.get_variable("biases_o_tmp", shape=[NUM_LABELS],
                                           initializer=tf.constant_initializer(0.0))

        logit_o_oblique = tf.matmul(fc1_o_oblique, fc2_weights_o) + fc2_biases_o

        return logit_o_oblique

