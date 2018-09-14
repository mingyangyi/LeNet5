import os
import tensorflow as tf

import LeNet5
import LeNet5_grassmann
import LeNet5_oblique

import optimize_function
import gutils
from tensorflow.examples.tutorials.mnist import input_data
import time

MODEL_SAVE_PATH="D:/paper/Two new methods on manifold optimization/code/output/tmp"
FILE_SAVE_PATH="D:/paper/Two new methods on manifold optimization/code/output"
MODEL_NAME="cnn_for_mnist"

BATCH_SIZE=100#批量大小
LEARNING_RATE_BASE=0.03#基础学习率，书中的基础学习率设置的太大了，高度非线性跳，导致梯度会很大，学习率一定要小。
LEARNING_RATE_DECAY=0.99#衰减学习率
REGULARIZATION_RATE=0.0001#正则项损失系数
TRAINING_STEPS=1201#训练轮数
MOVING_AVERAGE_DECAY=0.99#滑动平均衰减率
GRAD_CLIP=1.0
DELTA=tf.convert_to_tensor(0.001,dtype=tf.float32)

def train(mnist,LEARNING_RATE_BASE,MODEL_SAVE_PATH,FILE_SAVE_PATH):
    file_path_loss_g = os.path.join(FILE_SAVE_PATH, ('loss_g_' + str(LEARNING_RATE_BASE) + '.txt'))
    file_path_loss_o = os.path.join(FILE_SAVE_PATH, ('loss_o_' + str(LEARNING_RATE_BASE) + '.txt'))

    file_path_norm = os.path.join(FILE_SAVE_PATH, ('norm' + str(LEARNING_RATE_BASE) + '.txt'))

    file1_path = os.path.join(FILE_SAVE_PATH, ('accuracy_' + str(LEARNING_RATE_BASE) + '.txt'))
    file1_path_g = os.path.join(FILE_SAVE_PATH, ('accuracy_g_' + str(LEARNING_RATE_BASE) + '.txt'))
    file1_path_o = os.path.join(FILE_SAVE_PATH, ('accuracy_o_' + str(LEARNING_RATE_BASE) + '.txt'))

    file_loss_g = open(file_path_loss_g, 'w')
    file_loss_o = open(file_path_loss_o, 'w')

    file_norm=open(file_path_norm,'w')

    file_accuracy = open(file1_path, 'w')
    file_accuracy_g= open(file1_path_g, 'w')
    file_accuracy_o = open(file1_path_o, 'w')

    x=tf.placeholder(tf.float32,shape=[None,LeNet5.INPUT_NODE],name="x-input")
    y_=tf.placeholder(tf.float32,shape=[None,LeNet5.OUTPUT_NODE],name="y-output")
    x_reshaped=tf.reshape(x,[-1,LeNet5.IMAGE_SIZE,LeNet5.IMAGE_SIZE,LeNet5.NUM_CHANNELS])
    times=tf.placeholder(tf.float32,shape=None,name="times")

    #GRAD_CLIP=tf.constant(1.0,dtype=tf.float32)

    #正则化
    regularizer=tf.contrib.layers.l1_regularizer(REGULARIZATION_RATE)

    y_g,y_o = LeNet5.inference(x_reshaped,False,regularizer)
    global_step=tf.Variable(0,trainable=None)

    #定义损失函数，滑动平均操作等
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #variable_averages_op=variable_averages.apply(tf.trainable_variables())
    cross_entropy_g=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y_g)
    cross_entropy_o = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y_o)

    cross_entropy_mean_g = tf.reduce_mean(cross_entropy_g)
    cross_entropy_mean_o = tf.reduce_mean(cross_entropy_o)

    #损失函数，其中涉及到对一个列表中的元素（还是一个列表）求和
    loss_g=cross_entropy_mean_g#+tf.add_n(tf.get_collection('losses_g'))
    loss_o=cross_entropy_mean_o#+tf.add_n(tf.get_collection('losses_o'))

    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    #learning_rate=LEARNING_RATE_BASE
    #更新参数

    #train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #滑动平均并行计算
    #with tf.control_dependencies([train_step,variable_averages_op]):
        #train_op=tf.no_op(name='train')
    correct_prediction_g=tf.equal(tf.argmax(y_,1),tf.argmax(y_g,1))
    correct_prediction_o=tf.equal(tf.argmax(y_,1),tf.argmax(y_o,1))

    correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(tf.add(y_g,y_o),1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_g = tf.reduce_mean(tf.cast(correct_prediction_g, tf.float32))
    accuracy_o = tf.reduce_mean(tf.cast(correct_prediction_o, tf.float32))
#########################################################################################################3
    with tf.variable_scope('layer1-conv1', reuse=True):
        conv1_weights_g = tf.get_variable("weight_g")
        conv1_biases_g = tf.get_variable('biases_g')

        conv1_weights_o = tf.get_variable("weight_o")
        conv1_biases_o = tf.get_variable('biases_o')

        conv1_weights_g_tmp_layer1 = tf.get_variable("weight_g_tmp")
        conv1_weights_o_tmp_layer1 = tf.get_variable("weight_o_tmp")

        conv1_biases_g_tmp = tf.get_variable("biases_g_tmp")
        conv1_biases_o_tmp = tf.get_variable("biases_o_tmp")

        dim_layer1 = conv1_weights_g.get_shape()

        weights_grad_g_base_layer1 = tf.gradients(loss_g, conv1_weights_g, stop_gradients=conv1_weights_g)
        weights_grad_o_base_layer1 = tf.gradients(loss_o, conv1_weights_o, stop_gradients=conv1_weights_o)

        weights_grad_g_base_biases_layer1 = tf.gradients(loss_g, conv1_biases_g, stop_gradients=conv1_biases_g)
        weights_grad_o_base_biases_layer1 = tf.gradients(loss_o, conv1_biases_o, stop_gradients=conv1_biases_o)

        weights_g_1= tf.reshape(conv1_weights_g, shape=[-1, 1])
        weights_o_1 = tf.reshape(conv1_weights_o, shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_g_base_layer1[0], dtype=tf.float32)
        tf.convert_to_tensor(weights_grad_o_base_layer1[0], dtype=tf.float32)

        weights_grad_g_base_1 = tf.reshape(weights_grad_g_base_layer1[0], shape=[-1, 1])
        weights_grad_o_base_l = tf.reshape(weights_grad_o_base_layer1[0], shape=[-1, 1])

        grad_on_grassmann_1 = gutils.grassmann_project(weights_g_1, weights_grad_g_base_1)
        grad_on_oblique_1 = gutils.oblique_project(weights_o_1, weights_grad_o_base_l)

        weights_g_layer1 = optimize_function.apply_dense_on_grasssmann(GRAD_CLIP, grad_on_grassmann_1,
                                                                grad_on_oblique_1
                                                                , weights_g_1, learning_rate, times,
                                                                DELTA)

        weights_o_layer1 = optimize_function._apply_dense_on_oblique(GRAD_CLIP, grad_on_grassmann_1,
                                                              grad_on_oblique_1
                                                              , weights_o_1, learning_rate, times,
                                                              DELTA)

        weights_biases_g_layer1 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_g_base_biases_layer1[0], tf.float32), conv1_biases_g)
        weights_biases_o_layer1 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_o_base_biases_layer1[0], tf.float32), conv1_biases_o)

        norm_g_1 = tf.square(gutils.norm(grad_on_grassmann_1))
        norm_o_1 = tf.square(gutils.norm(grad_on_oblique_1))

    with tf.variable_scope('layer3-conv2', reuse=True):
        conv2_weights_g = tf.get_variable("weight_g")
        conv2_biases_g = tf.get_variable('biases_g')
        conv2_weights_o = tf.get_variable("weight_o")
        conv2_biases_o = tf.get_variable('biases_o')

        conv2_weights_g_tmp_layer3 = tf.get_variable("weight_g_tmp")
        conv2_weights_o_tmp_layer3 = tf.get_variable("weight_o_tmp")

        conv2_biases_g_tmp = tf.get_variable("biases_g_tmp")
        conv2_biases_o_tmp = tf.get_variable("biases_o_tmp")

        dim_layer3 = conv2_weights_g.get_shape()

        weights_grad_g_base_3 = tf.gradients(loss_g, conv2_weights_g,stop_gradients=conv2_weights_g)
        weights_grad_o_base_3 = tf.gradients(loss_o, conv2_weights_o,stop_gradients=conv2_weights_o)

        weights_grad_g_base_biases_layer3=tf.gradients(loss_g, conv2_biases_g, stop_gradients=conv2_biases_g)
        weights_grad_o_base_biases_layer3=tf.gradients(loss_o, conv2_biases_o, stop_gradients=conv2_biases_o)

        weights_g_3 = tf.reshape(conv2_weights_g, shape=[-1, 1])
        weights_o_3 = tf.reshape(conv2_weights_o, shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_g_base_3[0], dtype=tf.float32)
        tf.convert_to_tensor(weights_grad_o_base_3[0], dtype=tf.float32)

        weights_grad_g_3 = tf.reshape(weights_grad_g_base_3[0], shape=[-1, 1])
        weights_grad_o_3 = tf.reshape(weights_grad_o_base_3[0], shape=[-1, 1])

        grad_on_grassmann_3 = gutils.grassmann_project(weights_g_3, weights_grad_g_3)
        grad_on_oblique_3 = gutils.oblique_project(weights_o_3, weights_grad_o_3)

        weights_g_layer3 = optimize_function.apply_dense_on_grasssmann(GRAD_CLIP, grad_on_grassmann_3,
                                                                grad_on_oblique_3
                                                                , weights_g_3, learning_rate, times,
                                                                DELTA)
        weights_o_layer3 = optimize_function._apply_dense_on_oblique(GRAD_CLIP, grad_on_grassmann_3,
                                                              grad_on_oblique_3,
                                                              weights_o_3, learning_rate, times,
                                                              DELTA)

        weights_biases_g_layer3 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_g_base_biases_layer3[0], tf.float32), conv2_biases_g)
        weights_biases_o_layer3 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_o_base_biases_layer3[0], tf.float32), conv2_biases_o)

        norm_g_3 = tf.square(gutils.norm(grad_on_grassmann_3))
        norm_o_3=tf.square(gutils.norm(grad_on_oblique_3))

    with tf.variable_scope('layer5-fc1', reuse=True):
        fc1_weights_g = tf.get_variable("weight_g")
        fc1_biases_g = tf.get_variable("biases_g")
        fc1_weights_o = tf.get_variable("weight_o")
        fc1_biases_o = tf.get_variable("biases_o")

        fc1_weights_g_tmp_layer5 = tf.get_variable("weight_g_tmp")
        fc1_weights_o_tmp_layer5 = tf.get_variable("weight_o_tmp")

        fc1_biases_g_tmp = tf.get_variable("biases_g_tmp")
        fc1_biases_o_tmp = tf.get_variable("biases_o_tmp")

        dim_layer5 = fc1_weights_g.get_shape()

        weights_grad_g_base_5 = tf.gradients(loss_g, fc1_weights_g,stop_gradients=fc1_weights_g)
        weights_grad_o_base_5 = tf.gradients(loss_o, fc1_weights_o,stop_gradients=fc1_weights_o)

        weights_grad_g_base_biases_layer5 = tf.gradients(loss_g, fc1_biases_g, stop_gradients=fc1_biases_g)
        weights_grad_o_base_biases_layer5 = tf.gradients(loss_o, fc1_biases_o, stop_gradients=fc1_biases_o)

        weights_g_5 = tf.reshape(fc1_weights_g, shape=[-1, 1])
        weights_o_5 = tf.reshape(fc1_weights_o, shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_g_base_5[0], dtype=tf.float32)
        tf.convert_to_tensor(weights_grad_o_base_5[0], dtype=tf.float32)

        weights_grad_g_5 = tf.reshape(weights_grad_g_base_5[0], shape=[-1, 1])
        weights_grad_o_5 = tf.reshape(weights_grad_o_base_5[0], shape=[-1, 1])

        grad_on_grassmann_5 = gutils.grassmann_project(weights_g_5, weights_grad_g_5)
        grad_on_oblique_5 = gutils.oblique_project(weights_o_5, weights_grad_o_5)

        weights_g_layer5 = optimize_function.apply_dense_on_grasssmann(GRAD_CLIP, grad_on_grassmann_5,
                                                                grad_on_oblique_5
                                                                , weights_g_5, learning_rate, times, DELTA)
        weights_o_layer5 = optimize_function._apply_dense_on_oblique(GRAD_CLIP, grad_on_grassmann_5,
                                                              grad_on_oblique_5
                                                              , weights_o_5, learning_rate, times, DELTA)

        weights_biases_g_layer5 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_g_base_biases_layer5[0], tf.float32), fc1_biases_g)
        weights_biases_o_layer5 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_o_base_biases_layer5[0], tf.float32), fc1_biases_o)

        norm_g_5 = tf.square(gutils.norm(grad_on_grassmann_5))
        norm_o_5 = tf.square(gutils.norm(grad_on_oblique_5))

    with tf.variable_scope('layer6-fc2', reuse=True):
        fc2_weights_g = tf.get_variable("weight_g")
        fc2_biases_g = tf.get_variable("biases_g")
        fc2_weights_o = tf.get_variable("weight_o")
        fc2_biases_o = tf.get_variable("biases_o")

        fc2_weights_g_tmp_layer6 = tf.get_variable("weight_g_tmp")
        fc2_weights_o_tmp_layer6= tf.get_variable("weight_o_tmp")

        fc2_biases_g_tmp = tf.get_variable("biases_g_tmp")
        fc2_biases_o_tmp = tf.get_variable("biases_o_tmp")

        dim_layer6 = fc2_weights_g.get_shape()

        weights_grad_g_base_6 = tf.gradients(loss_g, fc2_weights_g)
        weights_grad_o_base_6 = tf.gradients(loss_o, fc2_weights_o)

        weights_grad_g_base_biases_layer6 = tf.gradients(loss_g, fc2_biases_g, stop_gradients=fc2_biases_g)
        weights_grad_o_base_biases_layer6 = tf.gradients(loss_o, fc2_biases_o, stop_gradients=fc2_biases_o)

        weights_g_6 = tf.reshape(fc2_weights_g, shape=[-1, 1])
        weights_o_6 = tf.reshape(fc2_weights_o, shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_g_base_6[0], dtype=tf.float32)
        tf.convert_to_tensor(weights_grad_o_base_6[0], dtype=tf.float32)

        weights_grad_g = tf.reshape(weights_grad_g_base_6[0], shape=[-1, 1])
        weights_grad_o = tf.reshape(weights_grad_o_base_6[0], shape=[-1, 1])

        grad_on_grassmann_6 = gutils.grassmann_project(weights_g_6, weights_grad_g)
        grad_on_oblique_6 = gutils.oblique_project(weights_o_6, weights_grad_o)

        weights_g_layer6 = optimize_function.apply_dense_on_grasssmann(GRAD_CLIP, grad_on_grassmann_6,
                                                                grad_on_oblique_6
                                                                , weights_g_6, learning_rate, times, DELTA)
        weights_o_layer6 = optimize_function._apply_dense_on_oblique(GRAD_CLIP, grad_on_grassmann_6,
                                                              grad_on_oblique_6
                                                              , weights_o_6, learning_rate, times, DELTA)

        weights_biases_g_layer6 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_g_base_biases_layer6[0], tf.float32), fc2_biases_g)
        weights_biases_o_layer6 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_o_base_biases_layer6[0], tf.float32), fc2_biases_o)

        norm_g_6 = tf.square(gutils.norm(grad_on_grassmann_6))
        norm_o_6 = tf.square(gutils.norm(grad_on_oblique_6))

        _1 = tf.assign(conv1_weights_g_tmp_layer1, tf.reshape(weights_g_layer1, shape=dim_layer1))
        _2 = tf.assign(conv1_weights_o_tmp_layer1, tf.reshape(weights_o_layer1, shape=dim_layer1))
        _3 = tf.assign(conv2_weights_g_tmp_layer3, tf.reshape(weights_g_layer3, shape=dim_layer3))
        _4 = tf.assign(conv2_weights_o_tmp_layer3, tf.reshape(weights_o_layer3, shape=dim_layer3))
        _5 = tf.assign(fc1_weights_g_tmp_layer5, tf.reshape(weights_g_layer5, shape=dim_layer5))
        _6 = tf.assign(fc1_weights_o_tmp_layer5, tf.reshape(weights_o_layer5, shape=dim_layer5))
        _7 = tf.assign(fc2_weights_g_tmp_layer6, tf.reshape(weights_g_layer6, shape=dim_layer6))
        _8 = tf.assign(fc2_weights_o_tmp_layer6, tf.reshape(weights_o_layer6, shape=dim_layer6))

        _11 = tf.assign(conv1_biases_g_tmp, weights_biases_g_layer1)
        _12 = tf.assign(conv1_biases_o_tmp, weights_biases_o_layer1)
        _13 = tf.assign(conv2_biases_g_tmp, weights_biases_g_layer3)
        _14 = tf.assign(conv2_biases_o_tmp, weights_biases_o_layer3)
        _15 = tf.assign(fc1_biases_g_tmp, weights_biases_g_layer5)
        _16 = tf.assign(fc1_biases_o_tmp, weights_biases_o_layer5)
        _17 = tf.assign(fc2_biases_g_tmp, weights_biases_g_layer6)
        _18 = tf.assign(fc2_biases_o_tmp, weights_biases_o_layer6)

        _21 = tf.assign(conv1_weights_g, conv1_weights_g_tmp_layer1)
        _22 = tf.assign(conv1_weights_o, conv1_weights_o_tmp_layer1)
        _23 = tf.assign(conv2_weights_g, conv2_weights_g_tmp_layer3)
        _24 = tf.assign(conv2_weights_o, conv2_weights_o_tmp_layer3)
        _25 = tf.assign(fc1_weights_g, fc1_weights_g_tmp_layer5)
        _26 = tf.assign(fc1_weights_o, fc1_weights_o_tmp_layer5)
        _27 = tf.assign(fc2_weights_g, fc2_weights_g_tmp_layer6)
        _28 = tf.assign(fc2_weights_o, fc2_weights_o_tmp_layer6)

        _31 = tf.assign(conv1_biases_g, conv1_biases_g_tmp)
        _32 = tf.assign(conv1_biases_o, conv1_biases_o_tmp)
        _33 = tf.assign(conv2_biases_g, conv2_biases_g_tmp)
        _34 = tf.assign(conv2_biases_o, conv2_biases_o_tmp)
        _35 = tf.assign(fc1_biases_g, fc1_biases_g_tmp)
        _36 = tf.assign(fc1_biases_o, fc1_biases_o_tmp)
        _37 = tf.assign(fc2_biases_g, fc2_biases_g_tmp)
        _38 = tf.assign(fc2_biases_o, fc2_biases_o_tmp)
######################################################################################################################
    #初始化持久化类
    #saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #训练模型，其中每隔一段时间会保存训练的结果
        for u in range(TRAINING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            loss_value_g, loss_value_o,\
            accuracy_value, accuracy_g_value, accuracy_o_value, step = sess.run(
                [loss_g, loss_o, accuracy, accuracy_g, accuracy_o,
                 global_step], feed_dict={x: xs, y_: ys})
#****************************************************************

            sess.run([_1, _2, _3, _4, _5, _6, _7, _8], feed_dict={x: xs, y_: ys, times: float(u)})
            sess.run([_11, _12, _13, _14, _15, _16, _17, _18], feed_dict={x: xs, y_: ys, times: float(u)})
            sess.run([_21, _22, _23, _24, _25, _26, _27, _28])
            sess.run([_31, _32, _33, _34, _35, _36, _37, _38])
##########################################################################################################
            file_loss_g.write(str(u)), file_loss_g.write(' '), file_loss_g.write(str(loss_value_g)), file_loss_g.write(
                "\n")
            file_loss_o.write(str(u)), file_loss_o.write(' '), file_loss_o.write(str(loss_value_o)), file_loss_o.write(
                "\n")

            file_accuracy.write(str(u)), file_accuracy.write(' '), file_accuracy.write(
                str(accuracy_value)), file_accuracy.write('\n')
            file_accuracy_g.write(str(u)), file_accuracy_g.write(' '), file_accuracy_g.write(
                str(accuracy_g_value)), file_accuracy_g.write('\n')
            file_accuracy_o.write(str(u)), file_accuracy_o.write(' '), file_accuracy_o.write(
                str(accuracy_o_value)), file_accuracy_o.write('\n')
            #file_norm.write(str(u)), file_norm.write(' '), file_norm.write(
            #    str(n)), file_norm.write('\n')

            if u%100==0:
                print("After %d training steps, loss_g and loss_o on training batch is %g and %g accuracy is %g" % (
                u, loss_value_g, loss_value_o, accuracy_value))

                print("After %d training steps, accuracy_g and accuracy_o on training batch is %g and %g" % (
                u, accuracy_g_value , accuracy_o_value))
                print(time.localtime(time.time()))
                #model_name=MODEL_NAME+"_"+str(LEARNING_RATE_BASE)+".ckpt"
                #saver.save(sess,os.path.join(MODEL_SAVE_PATH,model_name),global_step=global_step)
        xs = mnist.validation.images
        ys = mnist.validation.labels
        loss_value_g, loss_value_o, accuracy_value = sess.run(
            [loss_g, loss_o, accuracy], feed_dict={x: xs, y_: ys})

        print("The loss_g, loss_o and accuracy on validation is %g %g and %g" % (
        loss_value_g, loss_value_o, accuracy_value))


def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
    train(mnist,LEARNING_RATE_BASE,MODEL_SAVE_PATH,FILE_SAVE_PATH)

if __name__=='__main__':
    tf.app.run()