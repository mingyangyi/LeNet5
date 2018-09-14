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
CONV1_SIZE=10#卷积层尺寸
#第二层卷积的尺寸
CONV2_DEEP=64
CONV2_SIZE=10
#全连接层节点个数
FC_SIZE=512
#定义训练的前向传播，其中加入了dropout过程
def inference(input_tensor,train,regularizer):
    #第一层卷积层
    with tf.variable_scope('layer1-conv1'):
        input_tensor = batch_norm.batch_norm(input_tensor)

        conv1_weights_g=tf.get_variable("weight_g",shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1,seed=9))
        conv1_biases_g=tf.get_variable("biases_g",shape=[CONV1_DEEP],initializer=tf.constant_initializer(0.0))

        conv1_weights_o = tf.get_variable("weight_o", shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1,seed=21))

        #tf.assign(conv1_weights_o , gutils.unit(conv1_weights_o))
        #tf.assign(conv1_weights_g, gutils.unit(conv1_weights_g))

        conv1_biases_o=tf.get_variable("biases_o",shape=[CONV1_DEEP],initializer=tf.constant_initializer(0.0))

        conv1_weights_g_tmp=tf.get_variable("weight_g_tmp",shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=1))
        conv1_weights_o_tmp=tf.get_variable("weight_o_tmp",shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=1))

        conv1_biases_g_tmp = tf.get_variable("biases_g_tmp", shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.1))
        conv1_biases_o_tmp = tf.get_variable("biases_o_tmp", shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.1))

    #卷积网络前向传播，这里步长为1且做全0填充，输出是28*28*32的矩阵，步幅就在第二个参数矩阵里面了。
        conv1_g=tf.nn.conv2d(input_tensor,conv1_weights_g,strides=[1,1,1,1],padding='SAME')
        conv1_batch_g=batch_norm.batch_norm(conv1_g,scale=None)
        relu1_g=tf.nn.relu(tf.nn.bias_add(conv1_batch_g,conv1_biases_g))

        conv1_o = tf.nn.conv2d(input_tensor, conv1_weights_o, strides=[1, 1, 1, 1], padding='SAME')
        conv1_batch_o = batch_norm.batch_norm(conv1_o, scale=None)
        relu1_o = tf.nn.relu(tf.nn.bias_add(conv1_batch_o, conv1_biases_o))

    #第二层池化层，步长为2，全0填充，过滤器边长为2
    with tf.name_scope('layer2-pool1'):
        pool1_g=tf.nn.max_pool(relu1_g,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')#第二个参数是步幅，第三个参数是步长
        pool1_batch_g=batch_norm.batch_norm(pool1_g,scale=None)

        pool1_o = tf.nn.max_pool(relu1_o, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 第二个参数是步幅，第三个参数是步长
        pool1_batch_o = batch_norm.batch_norm(pool1_o, scale=None)
    #输出是14*14*32，池化不改变层数
    #第三层卷积层，步幅为5，步长为1，深度为64，全0填充，输出为14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights_g=tf.get_variable('weight_g',shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1,seed=271))
        conv2_biases_g=tf.get_variable('biases_g',shape=[CONV2_DEEP],initializer=tf.constant_initializer(0.1))

        conv2_weights_o = tf.get_variable('weight_o', shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1,seed=140))
        #tf.assign(conv2_weights_o , gutils.unit(conv2_weights_o))
        #tf.assign(conv2_weights_g, gutils.unit(conv2_weights_g))

        conv2_biases_o = tf.get_variable('biases_o', shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.1))

        conv2_weights_g_tmp = tf.get_variable("weight_g_tmp", shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                              initializer=tf.truncated_normal_initializer(stddev=1))
        conv2_weights_o_tmp = tf.get_variable("weight_o_tmp", shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                              initializer=tf.truncated_normal_initializer(stddev=1))

        conv2_biases_g_tmp = tf.get_variable('biases_g_tmp', shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2_biases_o_tmp = tf.get_variable('biases_o_tmp', shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))

    #卷积网络前向传播
        conv2_g=tf.nn.conv2d(pool1_batch_g,conv2_weights_g,strides=[1,1,1,1],padding='SAME')
        conv2_batch_g=batch_norm.batch_norm(conv2_g,scale=None)
        relu2_g=tf.nn.relu(tf.nn.bias_add(conv2_batch_g,conv2_biases_g))

        conv2_o = tf.nn.conv2d(pool1_batch_o, conv2_weights_o, strides=[1, 1, 1, 1], padding='SAME')
        conv2_batch_o = batch_norm.batch_norm(conv2_o, scale=None)
        relu2_o = tf.nn.relu(tf.nn.bias_add(conv2_batch_o, conv2_biases_o))

    #第四层池化层和第二层结构相同
    with tf.name_scope('layer4-pool2'):
        pool2_g = tf.nn.max_pool(relu2_g, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 第二个参数是步幅，第三个参数是步长
        pool2_batch_g=batch_norm.batch_norm(pool2_g,scale=None)

        pool2_o = tf.nn.max_pool(relu2_o, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 第二个参数是步幅，第三个参数是步长
        pool2_batch_o = batch_norm.batch_norm(pool2_o, scale=None)

    #第五层是引入dropout的全连接层
    #  输出是7*7*64
    # 第五层是全连接网络，输出节点是512个
    with tf.variable_scope('layer5-fc1'):
        pool_shape = pool2_batch_g.get_shape().as_list()
        # pool_shape的第一个数据pool_shape[0]就是batch的大小
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        # 重新改变输入的结构把它拉成一个向量做全连接
        reshaped_g = tf.reshape(pool2_batch_g, [-1, nodes])
        reshaped_o = tf.reshape(pool2_batch_o, [-1, nodes])
        fc1_weights_g=tf.get_variable('weight_g',
                                      shape=[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1,seed=21))
        fc1_weights_o = tf.get_variable('weight_o', shape=[nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1,seed=61))

        #tf.assign(fc1_weights_o , gutils.unit(fc1_weights_o))
        #tf.assign(fc1_weights_g, gutils.unit(fc1_weights_g))

        fc1_weights_g_tmp = tf.get_variable('weight_g_tmp',
                                        shape=[nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.01))
        fc1_weights_o_tmp = tf.get_variable('weight_o_tmp',
                                            shape=[nodes, FC_SIZE],
                                            initializer=tf.truncated_normal_initializer(stddev=0.01))

    #只对全连接参数做正则化
        if regularizer!=None:
            tf.add_to_collection('losses_g',regularizer(fc1_weights_g))
            tf.add_to_collection('losses_o', regularizer(fc1_weights_o))
        fc1_biases_g=tf.get_variable('biases_g',shape=[FC_SIZE],initializer=tf.constant_initializer(0.01))
        fc1_biases_o = tf.get_variable('biases_o', shape=[FC_SIZE], initializer=tf.constant_initializer(0.01))

        fc1_biases_g_tmp = tf.get_variable('biases_g_tmp', shape=[FC_SIZE], initializer=tf.constant_initializer(0.0))
        fc1_biases_o_tmp = tf.get_variable('biases_o_tmp', shape=[FC_SIZE], initializer=tf.constant_initializer(0.0))

        fc1_g = tf.nn.relu(tf.matmul(reshaped_g, fc1_weights_g) + fc1_biases_g)
        fc1_o = tf.nn.relu(tf.matmul(reshaped_o, fc1_weights_o) + fc1_biases_o)

        if train:
            fc1_g = tf.nn.dropout(fc1_g, 0.5)
            fc1_o = tf.nn.dropout(fc1_o, 0.5)
    #第六层输出层
    with tf.variable_scope('layer6-fc2'):
        fc2_weights_g = tf.get_variable('weight_g', shape=[FC_SIZE, NUM_LABELS],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1, seed=4))
        fc2_weights_o = tf.get_variable('weight_o', shape=[FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1,seed=214))
        #tf.assign(fc2_weights_o , gutils.unit(fc2_weights_o))
        #tf.assign(fc2_weights_g, gutils.unit(fc2_weights_g))

        fc2_weights_g_tmp = tf.get_variable('weight_g_tmp',
                                            shape=[FC_SIZE,NUM_LABELS],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_weights_o_tmp = tf.get_variable('weight_o_tmp',
                                            shape=[FC_SIZE,NUM_LABELS],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer!=None:
            tf.add_to_collection('losses_g',regularizer(fc2_weights_g))
            tf.add_to_collection('losses_o', regularizer(fc2_weights_o))

        fc2_biases_g = tf.get_variable('biases_g',shape=[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        fc2_biases_o = tf.get_variable('biases_o', shape=[NUM_LABELS], initializer=tf.constant_initializer(0.1))

        fc2_biases_g_tmp = tf.get_variable('biases_g_tmp', shape=[NUM_LABELS], initializer=tf.constant_initializer(0.0))
        fc2_biases_o_tmp = tf.get_variable('biases_o_tmp', shape=[NUM_LABELS], initializer=tf.constant_initializer(0.0))

        logit_g = tf.matmul(fc1_g, fc2_weights_g) + fc2_biases_g
        logit_o = tf.matmul(fc1_o, fc2_weights_o) + fc2_biases_o

        return logit_g,logit_o



