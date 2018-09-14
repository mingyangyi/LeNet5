import tensorflow as tf
import batch_norm

INPUT_NODE=784#输入节点数，就是图像的大小
OUTPUT_NODE=10#输出节点数，就是多分类问题
IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10
#第一卷积层的尺寸
CONV1_DEEP=32#卷积层深度
CONV1_SIZE=5#卷积层尺寸
#第二层卷积的尺寸
CONV2_DEEP=64
CONV2_SIZE=5
#全连接层节点个数
FC_SIZE=512
#定义训练的前向传播，其中加入了dropout过程

def inference(input_tensor,train,regularizer):

    with tf.variable_scope('layer1-conv1_grassmann'):
        conv1_weights_g = tf.get_variable("weight_g", shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1,seed=1))
        conv1_biases_g = tf.get_variable("biases_g", shape=[CONV1_DEEP], initializer=tf.constant_initializer(0))

        conv1_weights_g_tmp = tf.get_variable("weight_g_tmp", shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                              initializer=tf.truncated_normal_initializer(stddev=1))

        conv1_biases_g_tmp = tf.get_variable("biases_g_tmp", shape=[CONV1_DEEP],
                                             initializer=tf.constant_initializer(0.0))

    #卷积网络前向传播，这里步长为1且做全0填充，输出是28*28*32的矩阵，步幅就在第二个参数矩阵里面了。
        conv1_g=tf.nn.conv2d(input_tensor,conv1_weights_g,strides=[1,1,1,1],padding='SAME')
        conv1_batch_g=batch_norm.batch_norm(conv1_g,scale=None)
        relu1_g_grassmann=tf.nn.relu(tf.nn.bias_add(conv1_batch_g,conv1_biases_g))

    with tf.name_scope('layer2-pool1_grassmann'):
        pool1_g=tf.nn.max_pool(relu1_g_grassmann,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#第二个参数是步幅，第三个参数是步长
        pool1_batch_g_grassmann=batch_norm.batch_norm(pool1_g,scale=None)

    with tf.variable_scope('layer3-conv2_grassmann'):
        conv2_weights_g=tf.get_variable('weight_g',shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1,seed=3))
        conv2_biases_g=tf.get_variable('biases_g',shape=[CONV2_DEEP],initializer=tf.constant_initializer(0.0))

        conv2_weights_g_tmp = tf.get_variable("weight_g_tmp", shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                              initializer=tf.truncated_normal_initializer(stddev=1))

        conv2_biases_g_tmp = tf.get_variable("biases_g_tmp", shape=[CONV2_DEEP],
                                             initializer=tf.constant_initializer(0.0))
    #卷积网络前向传播
        conv2_g=tf.nn.conv2d(pool1_batch_g_grassmann,conv2_weights_g,strides=[1,1,1,1],padding='SAME')
        conv2_batch_g=batch_norm.batch_norm(conv2_g,scale=None)
        relu2_g_grassmann=tf.nn.relu(tf.nn.bias_add(conv2_batch_g,conv2_biases_g))

    with tf.name_scope('layer4-pool2_grassmann'):
        pool2_g = tf.nn.max_pool(relu2_g_grassmann, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 第二个参数是步幅，第三个参数是步长
        pool2_batch_g_grassmann=batch_norm.batch_norm(pool2_g,scale=None)

    with tf.variable_scope('layer5-fc1_grassmann'):
        pool_shape = pool2_batch_g_grassmann.get_shape().as_list()
        # pool_shape的第一个数据pool_shape[0]就是batch的大小
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        # 重新改变输入的结构把它拉成一个向量做全连接
        reshaped_g = tf.reshape(pool2_batch_g_grassmann, [-1, nodes])
        fc1_weights_g=tf.get_variable('weight_g',
                                      shape=[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1,seed=5))

        fc1_weights_g_tmp = tf.get_variable('weight_g_tmp',
                                        shape=[nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))

    #只对全连接参数做正则化
        if regularizer!=None:
            tf.add_to_collection('losses_g_grassmann',regularizer(fc1_weights_g))
        fc1_biases_g = tf.get_variable('biases_g',shape=[FC_SIZE],initializer=tf.constant_initializer(0.0))

        fc1_biases_g_tmp = tf.get_variable("biases_g_tmp", shape=[FC_SIZE],
                                           initializer=tf.constant_initializer(0.0))

        fc1_g_grassmann = tf.nn.relu(tf.matmul(reshaped_g, fc1_weights_g) + fc1_biases_g)

        if train:
            fc1_g_grassmann=tf.nn.dropout(fc1_g_grassmann,0.5)

    with tf.variable_scope('layer6-fc2_grassmann'):
        fc2_weights_g=tf.get_variable('weight_g',shape=[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1,seed=5))

        fc2_weights_g_tmp = tf.get_variable('weight_g_tmp',
                                            shape=[FC_SIZE,NUM_LABELS],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer!=None:
            tf.add_to_collection('losses_g_grassmann',regularizer(fc2_weights_g))

        fc2_biases_g = tf.get_variable('biases_g',shape=[NUM_LABELS],initializer=tf.constant_initializer(0.0))

        fc2_biases_g_tmp = tf.get_variable("biases_g_tmp", shape=[NUM_LABELS],
                                           initializer=tf.constant_initializer(0.0))

        logit_g_grassmann = tf.matmul(fc1_g_grassmann,fc2_weights_g)+fc2_biases_g

        return logit_g_grassmann

