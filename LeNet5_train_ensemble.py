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
LEARNING_RATE_BASE=0.1#基础学习率，书中的基础学习率设置的太大了，高度非线性跳，导致梯度会很大，学习率一定要小。
LEARNING_RATE_OBLIQUE=0.1
LEARNING_RATE_DECAY=0.99#衰减学习率
REGULARIZATION_RATE=0.001#正则项损失系数
TRAINING_STEPS=601#训练轮数
MOVING_AVERAGE_DECAY=0.99#滑动平均衰减率
GRAD_CLIP=1.0
DELTA=tf.convert_to_tensor(0.001,dtype=tf.float32)

def train(mnist,LEARNING_RATE_BASE,MODEL_SAVE_PATH,FILE_SAVE_PATH):
    file_path_loss_grassmann = os.path.join(FILE_SAVE_PATH, ('loss_grassmann_' + str(LEARNING_RATE_BASE) + '.txt'))
    file_path_loss_oblique = os.path.join(FILE_SAVE_PATH, ('loss_oblique_' + str(LEARNING_RATE_OBLIQUE) + '.txt'))

    file1_path_grassmann = os.path.join(FILE_SAVE_PATH, ('accuracy_grassmann_' + str(LEARNING_RATE_BASE) + '.txt'))
    file1_path_oblique = os.path.join(FILE_SAVE_PATH, ('accuracy_oblique' + str(LEARNING_RATE_OBLIQUE) + '.txt'))
    file1_path_ensemble = os.path.join(FILE_SAVE_PATH, ('accuracy_ensemble' + str(LEARNING_RATE_BASE) + '.txt'))

    file_path_norm_grassmann = os.path.join(FILE_SAVE_PATH, ('norm_grassmann' + str(LEARNING_RATE_BASE) + '.txt'))
    file_path_norm_oblique = os.path.join(FILE_SAVE_PATH, ('norm_oblique' + str(LEARNING_RATE_OBLIQUE) + '.txt'))

    file_loss_grassmann = open(file_path_loss_grassmann, 'w')
    file_loss_oblique = open(file_path_loss_oblique, 'w')

    file_accuracy_grassmann = open(file1_path_grassmann, 'w')
    file_accuracy_oblique = open(file1_path_oblique, 'w')
    file_accuracy_ensemble = open(file1_path_ensemble, 'w')

    file_norm_grassmann = open(file_path_norm_grassmann, 'w')
    file_norm_oblique = open(file_path_norm_oblique, 'w')

    x=tf.placeholder(tf.float32,shape=[None,LeNet5.INPUT_NODE],name="x-input")
    y_=tf.placeholder(tf.float32,shape=[None,LeNet5.OUTPUT_NODE],name="y-output")
    x_reshaped=tf.reshape(x,[-1,LeNet5.IMAGE_SIZE,LeNet5.IMAGE_SIZE,LeNet5.NUM_CHANNELS])
    times=tf.placeholder(tf.float32,shape=None,name="times")

    #GRAD_CLIP=tf.constant(1.0,dtype=tf.float32)

    #正则化
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y_g_grassmann = LeNet5_grassmann.inference(x_reshaped,False,regularizer)
    y_o_oblique = LeNet5_oblique.inference(x_reshaped,False,regularizer)
    global_step=tf.Variable(0,trainable=None)

    #定义损失函数，滑动平均操作等
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    cross_entropy_g_grassmann = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y_g_grassmann)
    cross_entropy_o_oblique = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y_o_oblique)

    cross_entropy_mean_g_grassmann = tf.reduce_mean(cross_entropy_g_grassmann)
    cross_entropy_mean_o_oblique = tf.reduce_mean(cross_entropy_o_oblique)

    #损失函数，其中涉及到对一个列表中的元素（还是一个列表）求和

    loss_g_grassmann = cross_entropy_mean_g_grassmann #+ tf.add_n(tf.get_collection('losses_g_grassmann'))
    loss_o_oblique = cross_entropy_mean_o_oblique #+ tf.add_n(tf.get_collection('losses_o_oblique'))

    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    learning_rate_o = tf.train.exponential_decay(LEARNING_RATE_OBLIQUE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    #learning_rate=LEARNING_RATE_BASE
    #更新参数

    #train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #滑动平均并行计算
    #with tf.control_dependencies([train_step,variable_averages_op]):
        #train_op=tf.no_op(name='train')
    correct_prediction_grassmann = tf.equal(tf.argmax(y_, 1), tf.argmax(y_g_grassmann, 1))
    correct_prediction_oblique = tf.equal(tf.argmax(y_, 1), tf.argmax(y_o_oblique, 1))

    correct_prediction_ensemble=tf.equal(tf.argmax(y_,1),tf.argmax(tf.add(y_g_grassmann,y_o_oblique),1))

    accuracy_grassmann = tf.reduce_mean(tf.cast(correct_prediction_grassmann, tf.float32))
    accuracy_oblique = tf.reduce_mean(tf.cast(correct_prediction_oblique, tf.float32))
    accuracy_ensemble = tf.reduce_mean(tf.cast(correct_prediction_ensemble, tf.float32))
###########################################################################################################
    with tf.variable_scope('layer1-conv1_grassmann', reuse=True):
        conv1_weights_g = tf.get_variable("weight_g")
        conv1_biases_g = tf.get_variable('biases_g')

        conv1_weights_g_tmp_grassmann = tf.get_variable("weight_g_tmp")
        conv1_biases_g_tmp_grassmann = tf.get_variable("biases_g_tmp")

        weights_grad_g_base_g_layer1 = tf.gradients(loss_g_grassmann, conv1_weights_g, stop_gradients=conv1_weights_g)

        weights_grad_g_base_g_biases_layer1 = tf.gradients(loss_g_grassmann, conv1_biases_g, stop_gradients=conv1_biases_g)

        weights_g = tf.reshape(conv1_weights_g , shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_g_base_g_layer1[0], dtype=tf.float32)

        weights_grad_g = tf.reshape(weights_grad_g_base_g_layer1[0], shape=[-1, 1])

        grad_on_grassmann = gutils.grassmann_project(weights_g, weights_grad_g)

        weights_g_layer1 = optimize_function._apply_dense_on_grassmann_with_noise(GRAD_CLIP, grad_on_grassmann,
                                                                  weights_g, 100,learning_rate, times)
        #weights_g_layer1=weights_g-learning_rate*weights_grad_g

        weights_biases_grassmann_layer1 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_g_base_g_biases_layer1[0], tf.float32), conv1_biases_g)

        norm_g_1=tf.square(gutils.norm(grad_on_grassmann))

    with tf.variable_scope('layer3-conv2_grassmann', reuse=True):
        conv2_weights_g = tf.get_variable("weight_g")
        conv2_biases_g = tf.get_variable('biases_g')

        conv2_weights_g_tmp_grassmann = tf.get_variable("weight_g_tmp")
        conv2_biases_g_tmp_grassmann = tf.get_variable("biases_g_tmp")

        weights_grad_g_base_g_layer3 = tf.gradients(loss_g_grassmann, conv2_weights_g,stop_gradients=conv2_weights_g)

        weights_grad_g_base_g_biases_layer3 = tf.gradients(loss_g_grassmann, conv2_biases_g,
                                                           stop_gradients=conv2_biases_g)

        weights_g = tf.reshape(conv2_weights_g, shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_g_base_g_layer3[0], dtype=tf.float32)

        weights_grad_g = tf.reshape(weights_grad_g_base_g_layer3[0], shape=[-1, 1])

        grad_on_grassmann = gutils.grassmann_project(weights_g, weights_grad_g)

        weights_g_layer3 = optimize_function._apply_dense_on_grassmann_with_noise(GRAD_CLIP, grad_on_grassmann
                                                                  , weights_g, 101,learning_rate, times)

        #weights_g_layer3 = weights_g - learning_rate * weights_grad_g

        weights_biases_grassmann_layer3 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_g_base_g_biases_layer3[0], tf.float32),
            conv2_biases_g)
        norm_g_3 = tf.square(gutils.norm(grad_on_grassmann))

    with tf.variable_scope('layer5-fc1_grassmann', reuse=True):
            fc1_weights_g = tf.get_variable("weight_g")
            fc1_biases_g = tf.get_variable("biases_g")

            fc1_weights_g_tmp_grassmann = tf.get_variable("weight_g_tmp")
            fc1_biases_g_tmp_grassmann = tf.get_variable("biases_g_tmp")

            weights_grad_g_base_g_layer5 = tf.gradients(loss_g_grassmann, fc1_weights_g, stop_gradients=fc1_weights_g)

            weights_grad_g_base_biases_g_layer5 = tf.gradients(loss_g_grassmann, fc1_biases_g, stop_gradients=fc1_biases_g)

            weights_g = tf.reshape(fc1_weights_g, shape=[-1, 1])

            tf.convert_to_tensor(weights_grad_g_base_g_layer5[0], dtype=tf.float32)

            weights_grad_g = tf.reshape(weights_grad_g_base_g_layer5[0], shape=[-1, 1])

            grad_on_grassmann = gutils.grassmann_project(weights_g, weights_grad_g)

            weights_g_layer5 = optimize_function._apply_dense_on_grassmann_with_noise(GRAD_CLIP, grad_on_grassmann
                                                                      , weights_g, 102,learning_rate, times)

            #weights_g_layer5 = weights_g - learning_rate * weights_grad_g

            weights_biases_grassmann_layer5 = tf.add(
                -1 * learning_rate * tf.convert_to_tensor(weights_grad_g_base_biases_g_layer5[0], tf.float32),
                fc1_biases_g)
            norm_g_5 = tf.square(gutils.norm(grad_on_grassmann))

    with tf.variable_scope('layer6-fc2_grassmann', reuse=True):
        fc2_weights_g = tf.get_variable("weight_g")
        fc2_biases_g = tf.get_variable("biases_g")

        fc2_weights_g_tmp_grassmann = tf.get_variable("weight_g_tmp")
        fc2_biases_g_tmp_grassmann = tf.get_variable("biases_g_tmp")

        weights_grad_g_base_g_layer6 = tf.gradients(loss_g_grassmann, fc2_weights_g , stop_gradients=fc2_weights_g)

        weights_grad_g_base_biases_g_layer6= tf.gradients(loss_g_grassmann, fc2_biases_g, stop_gradients=fc2_biases_g)

        weights_g = tf.reshape(fc2_weights_g, shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_g_base_g_layer6[0], dtype=tf.float32)

        weights_grad_g = tf.reshape(weights_grad_g_base_g_layer6[0], shape=[-1, 1])

        grad_on_grassmann = gutils.grassmann_project(weights_g, weights_grad_g)

        weights_g_layer6 = optimize_function._apply_dense_on_grassmann_with_noise(GRAD_CLIP, grad_on_grassmann,
                                                                  weights_g,103, learning_rate, times)

        #weights_g_layer6 = weights_g-learning_rate*weights_grad_g

        weights_biases_grassmann_layer6 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_g_base_biases_g_layer6[0], tf.float32),
            fc2_biases_g)
        norm_g_6 = tf.square(gutils.norm(grad_on_grassmann))

############################################################################################################
    with tf.variable_scope('layer1-conv1_oblique', reuse=True):

        conv1_weights_o = tf.get_variable("weight_o")
        conv1_biases_o = tf.get_variable('biases_o')

        dim_layer1 = conv1_weights_o.get_shape()

        conv1_weights_o_tmp = tf.get_variable("weight_o_tmp")
        conv1_biases_o_tmp = tf.get_variable("biases_o_tmp")

        weights_grad_o_base_layer1_o = tf.gradients(loss_o_oblique, conv1_weights_o, stop_gradients=conv1_weights_o)

        weights_grad_o_base_biases_layer1_o = tf.gradients(loss_o_oblique, conv1_biases_o, stop_gradients=conv1_biases_o)

        weights_o = tf.reshape(conv1_weights_o, shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_o_base_layer1_o[0], dtype=tf.float32)

        weights_grad_o = tf.reshape(weights_grad_o_base_layer1_o[0], shape=[-1, 1])

        grad_on_oblique = gutils.oblique_project(weights_o, weights_grad_o)

        weights_o_layer1_o = optimize_function._apply_dense_on_oblique_with_noise(GRAD_CLIP, grad_on_oblique
                                                                , weights_o, 104,learning_rate_o, times)

        #weights_o_layer1_o = weights_o - learning_rate * weights_grad_o
        weights_biases_oblique_layer1 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_o_base_biases_layer1_o[0], tf.float32),
            conv1_biases_o)
        norm_o_1 = tf.square(gutils.norm(grad_on_oblique))

    with tf.variable_scope('layer3-conv2_oblique', reuse=True):
        conv2_weights_o = tf.get_variable("weight_o")
        conv2_biases_o = tf.get_variable('biases_o')

        conv2_weights_o_tmp = tf.get_variable("weight_o_tmp")
        conv2_biases_o_tmp = tf.get_variable("biases_o_tmp")

        dim_layer3=conv2_weights_o.get_shape()

        weights_grad_o_base_layer3_o = tf.gradients(loss_o_oblique, conv2_weights_o , stop_gradients=conv2_weights_o)

        weights_grad_o_base_biases_layer3_o = tf.gradients(loss_o_oblique, conv2_biases_o, stop_gradients=conv2_biases_o)

        weights_o = tf.reshape(conv2_weights_o, shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_o_base_layer3_o[0], dtype=tf.float32)

        weights_grad_o = tf.reshape(weights_grad_o_base_layer3_o[0], shape=[-1, 1])

        grad_on_oblique = gutils.oblique_project(weights_o, weights_grad_o)

        weights_o_layer3_o = optimize_function._apply_dense_on_oblique_with_noise(GRAD_CLIP, grad_on_oblique,
                                                                weights_o, 105,learning_rate_o, times)

        #weights_o_layer3_o = weights_o - learning_rate * weights_grad_o
        weights_biases_oblique_layer3 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_o_base_biases_layer3_o[0], tf.float32),
            conv2_biases_o)
        norm_o_3 = tf.square(gutils.norm(grad_on_oblique))

    with tf.variable_scope('layer5-fc1_oblique', reuse=True):
        fc1_weights_o = tf.get_variable("weight_o")
        fc1_biases_o = tf.get_variable("biases_o")

        fc1_weights_o_tmp = tf.get_variable("weight_o_tmp")
        fc1_biases_o_tmp = tf.get_variable("biases_o_tmp")

        dim_layer5 = fc1_weights_o.get_shape()

        weights_grad_o_base_layer5_o = tf.gradients(loss_o_oblique, fc1_weights_o ,stop_gradients=fc1_weights_o)

        weights_grad_o_base_biases_layer5_o = tf.gradients(loss_o_oblique, fc1_biases_o, stop_gradients=fc1_biases_o)

        weights_o = tf.reshape(fc1_weights_o, shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_o_base_layer5_o[0], dtype=tf.float32)

        weights_grad_o = tf.reshape(weights_grad_o_base_layer5_o[0], shape=[-1, 1])

        grad_on_oblique = gutils.oblique_project(weights_o, weights_grad_o)

        weights_o_layer5_o = optimize_function._apply_dense_on_oblique_with_noise(GRAD_CLIP, grad_on_oblique
                                                                , weights_o, 106,learning_rate_o, times)

        #weights_o_layer5_o = weights_o - learning_rate * weights_grad_o
        weights_biases_oblique_layer5 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_o_base_biases_layer5_o[0], tf.float32),
            fc1_biases_o)
        norm_o_5 = tf.square(gutils.norm(grad_on_oblique))

    with tf.variable_scope('layer6-fc2_oblique', reuse=True):
        fc2_weights_o = tf.get_variable("weight_o")
        fc2_biases_o = tf.get_variable("biases_o")

        fc2_weights_o_tmp = tf.get_variable("weight_o_tmp")
        fc2_biases_o_tmp = tf.get_variable("biases_o_tmp")

        dim_layer6 = fc2_weights_o.get_shape()

        weights_grad_o_base_layer6_o = tf.gradients(loss_o_oblique, fc2_weights_o , stop_gradients=fc2_weights_o)

        weights_grad_o_base_biases_layer6_o = tf.gradients(loss_o_oblique, fc2_biases_o, stop_gradients=fc2_biases_o)

        weights_o = tf.reshape(fc2_weights_o, shape=[-1, 1])

        tf.convert_to_tensor(weights_grad_o_base_layer6_o[0], dtype=tf.float32)

        weights_grad_o = tf.reshape(weights_grad_o_base_layer6_o[0], shape=[-1, 1])

        grad_on_oblique = gutils.oblique_project(weights_o, weights_grad_o)

        weights_o_layer6_o = optimize_function._apply_dense_on_oblique_with_noise(GRAD_CLIP, grad_on_oblique
                                                                , weights_o, 107,learning_rate_o, times)

        #weights_o_layer6_o = weights_o - learning_rate * weights_grad_o
        weights_biases_oblique_layer6 = tf.add(
            -1 * learning_rate * tf.convert_to_tensor(weights_grad_o_base_biases_layer6_o[0], tf.float32),
            fc2_biases_o)
        norm_o_6 = tf.square(gutils.norm(grad_on_oblique))
######################################################################################################################
        _1 = tf.assign(conv1_weights_g_tmp_grassmann, tf.reshape(weights_g_layer1, shape=dim_layer1))
        _2 = tf.assign(conv1_weights_o_tmp, tf.reshape(weights_o_layer1_o, shape=dim_layer1))
        _3 = tf.assign(conv2_weights_g_tmp_grassmann, tf.reshape(weights_g_layer3, shape=dim_layer3))
        _4 = tf.assign(conv2_weights_o_tmp, tf.reshape(weights_o_layer3_o, shape=dim_layer3))
        _5 = tf.assign(fc1_weights_g_tmp_grassmann, tf.reshape(weights_g_layer5, shape=dim_layer5))
        _6 = tf.assign(fc1_weights_o_tmp, tf.reshape(weights_o_layer5_o, shape=dim_layer5))
        _7 = tf.assign(fc2_weights_g_tmp_grassmann, tf.reshape(weights_g_layer6, shape=dim_layer6))
        _8 = tf.assign(fc2_weights_o_tmp, tf.reshape(weights_o_layer6_o, shape=dim_layer6))

        _11 = tf.assign(conv1_biases_g_tmp_grassmann, weights_biases_grassmann_layer1)
        _12 = tf.assign(conv1_biases_o_tmp, weights_biases_oblique_layer1)
        _13 = tf.assign(conv2_biases_g_tmp_grassmann, weights_biases_grassmann_layer3)
        _14 = tf.assign(conv2_biases_o_tmp, weights_biases_oblique_layer3)
        _15 = tf.assign(fc1_biases_g_tmp_grassmann, weights_biases_grassmann_layer5)
        _16 = tf.assign(fc1_biases_o_tmp, weights_biases_oblique_layer5)
        _17 = tf.assign(fc2_biases_g_tmp_grassmann, weights_biases_grassmann_layer6)
        _18 = tf.assign(fc2_biases_o_tmp, weights_biases_oblique_layer6)

        _21 = tf.assign(conv1_weights_g, conv1_weights_g_tmp_grassmann)
        _22 = tf.assign(conv1_weights_o, conv1_weights_o_tmp)
        _23 = tf.assign(conv2_weights_g, conv2_weights_g_tmp_grassmann)
        _24 = tf.assign(conv2_weights_o, conv2_weights_o_tmp)
        _25 = tf.assign(fc1_weights_g, fc1_weights_g_tmp_grassmann)
        _26 = tf.assign(fc1_weights_o, fc1_weights_o_tmp)
        _27 = tf.assign(fc2_weights_g, fc2_weights_g_tmp_grassmann)
        _28 = tf.assign(fc2_weights_o, fc2_weights_o_tmp)

        _31 = tf.assign(conv1_biases_g, conv1_biases_g_tmp_grassmann)
        _32 = tf.assign(conv1_biases_o, conv1_biases_o_tmp)
        _33 = tf.assign(conv2_biases_g, conv2_biases_g_tmp_grassmann)
        _34 = tf.assign(conv2_biases_o, conv2_biases_o_tmp)
        _35 = tf.assign(fc1_biases_g, fc1_biases_g_tmp_grassmann)
        _36 = tf.assign(fc1_biases_o, fc1_biases_o_tmp)
        _37 = tf.assign(fc2_biases_g, fc2_biases_g_tmp_grassmann)
        _38 = tf.assign(fc2_biases_o, fc2_biases_o_tmp)
    #初始化持久化类
    #saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #训练模型，其中每隔一段时间会保存训练的结果
        for u in range(TRAINING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            loss_value_g_grassmann,loss_value_o_oblique,\
            accuracy_g_grassmann_value,accuracy_o_oblique_value,accuracy_ensemble_value,\
            step=sess.run([loss_g_grassmann,loss_o_oblique,accuracy_grassmann,accuracy_oblique,accuracy_ensemble,
                           global_step],feed_dict={x:xs,y_:ys})

#****************************************************************
            sess.run([_1, _2, _3, _4, _5, _6, _7, _8], feed_dict={x: xs, y_: ys, times: float(u)})
            sess.run([_11, _12, _13, _14, _15, _16, _17, _18], feed_dict={x: xs, y_: ys, times: float(u)})
            sess.run([_21, _22, _23, _24, _25, _26, _27, _28])
            sess.run([_31, _32, _33, _34, _35, _36, _37, _38])

##########################################################################################################
            file_loss_grassmann.write(str(u)), file_loss_grassmann.write(' '), file_loss_grassmann.write(
                str(loss_value_g_grassmann)), file_loss_grassmann.write("\n")
            file_loss_oblique.write(str(u)), file_loss_oblique.write(' '), file_loss_oblique.write(
                str(loss_value_o_oblique)), file_loss_oblique.write("\n")

            file_accuracy_grassmann.write(str(u)), file_accuracy_grassmann.write(' '), file_accuracy_grassmann.write(
                str(accuracy_g_grassmann_value)), file_accuracy_grassmann.write('\n')
            file_accuracy_oblique.write(str(u)), file_accuracy_oblique.write(' '), file_accuracy_oblique.write(
                str(accuracy_o_oblique_value)), file_accuracy_oblique.write('\n')
            file_accuracy_ensemble.write(str(u)), file_accuracy_ensemble.write(' '), file_accuracy_ensemble.write(
                str(accuracy_ensemble_value)), file_accuracy_ensemble.write('\n')
            #file_norm_grassmann.write(str(u)), file_norm_grassmann.write(' '), file_norm_grassmann.write(
                #str(n_g)), file_norm_grassmann.write('\n')
            #file_norm_oblique.write(str(u)), file_norm_oblique.write(' '), file_norm_oblique.write(
            #    str(n_o)), file_norm_oblique.write('\n')

            if u%100==0:
                print("After %d training steps, accuracy_grassmann accuracy_oblique and accuracy_ensemble on training batch is %g , %g and %g" % (
                    u, accuracy_g_grassmann_value, accuracy_o_oblique_value, accuracy_ensemble_value ))
                print("After %d training steps, loss_g and loss_o on training batch is %g , %g" % (
                        u, loss_value_g_grassmann, loss_value_o_oblique))

                print(time.localtime(time.time()))
                #model_name=MODEL_NAME+"_"+str(LEARNING_RATE_BASE)+".ckpt"
                #saver.save(sess,os.path.join(MODEL_SAVE_PATH,model_name),global_step=global_step)
        xs = mnist.validation.images
        ys = mnist.validation.labels
        loss_value_g_grassmann, loss_value_o_oblique, accuracy_ensemble_value = sess.run(
            [loss_g_grassmann, loss_o_oblique, accuracy_ensemble], feed_dict={x: xs, y_: ys})

        print("The loss_g, loss_o and accuracy on validation is %g %g and %g" % (
        loss_value_g_grassmann, loss_value_o_oblique, accuracy_ensemble_value))


def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
    train(mnist,LEARNING_RATE_BASE,MODEL_SAVE_PATH,FILE_SAVE_PATH)

if __name__=='__main__':
    tf.app.run()