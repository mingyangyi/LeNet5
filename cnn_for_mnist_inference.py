import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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
    #第一层卷积层
    with tf.variable_scope('layer1-conv1'):
        conv1_weights=tf.get_variable("weight",shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("biases",shape=[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
    #卷积网络前向传播，这里步长为1且做全0填充，输出是28*28*32的矩阵，步幅就在第二个参数矩阵里面了。
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    #第二层池化层，步长为2，全0填充，过滤器边长为2
    with tf.name_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#第二个参数是步幅，第三个参数是步长
    #输出是14*14*32，池化不改变层数
    #第三层卷积层，步幅为5，步长为1，深度为64，全0填充，输出为14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights=tf.get_variable('weight',shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable('biases',shape=[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
    #卷积网络前向传播
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    #第四层池化层和第二层结构相同
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 第二个参数是步幅，第三个参数是步长
    #输出是7*7*64
    #第五层是全连接网络，输出节点是512个
    pool_shape=pool2.get_shape().as_list()
    #pool_shape的第一个数据pool_shape[0]就是batch的大小
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    #重新改变输入的结构把它拉成一个向量做全连接
    reshaped=tf.reshape(pool2,[-1,nodes])
    #第五层是引入dropout的全连接层
    with tf.variable_scope('layer5-fc1'):
        fc1_weights=tf.get_variable('weight',shape=[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
    #只对全连接参数做正则化
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable('biases',shape=[FC_SIZE],initializer=tf.constant_initializer(0.0))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:
            fc1=tf.nn.dropout(fc1,0.5)
    #第六层输出层
    with tf.variable_scope('layer6-fc2'):
        fc2_weights=tf.get_variable('weight',shape=[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('biases',shape=[NUM_LABELS],initializer=tf.constant_initializer(0.0))
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases
        return logit


x=tf.constant([[[1,2],[3,4]],[[10,100],[20,6]]],shape=[2,2,2],name="x",dtype=tf.float32)
z=tf.get_variable("z",shape=[1,28,28,1],dtype=tf.float32,initializer=tf.constant_initializer(1.0))
y=tf.constant(2,shape=[1],name="y",dtype=tf.float32)

with tf.Session() as sess:
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    xs, ys = mnist.train.next_batch(100)
    x_reshaped = tf.reshape(xs, [-1, IMAGE_SIZE, IMAGE_SIZE,
                                 NUM_CHANNELS])
    g = inference(x_reshaped, True, None)
    tf.global_variables_initializer().run()
    #print(sess.run(g))
    #print(sess.run(g))