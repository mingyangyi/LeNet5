import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import  LeNet5_train_ensemble
import LeNet5

EVAL_INTERVAL_SEC=10
#在测试集上看结果如何
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,LeNet5.INPUT_NODE],name='x-input')
        y_=tf.placeholder(tf.float32,[None,LeNet5.OUTPUT_NODE],name='y-input')
        x_reshaped=tf.reshape(x,[-1,LeNet5.IMAGE_SIZE,LeNet5.IMAGE_SIZE,LeNet5.NUM_CHANNELS])
        y=LeNet5.inference(x_reshaped,False,None)
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        #错误率
        correct_predict=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_predict,tf.float32))
        #重命名方式加载模型
        variable_averages=tf.train.ExponentialMovingAverage(LeNet5_train_ensemble.MOVING_AVERAGE_DECAY)
        variable_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variable_to_restore)
        while True:
            with tf.Session() as sess:
                #自动找到目录中的最新文件，那么就会是最后一个文件
                ckpt=tf.train.get_checkpoint_state(LeNet5_train_ensemble.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print("After %s trianing steps, validation accuracy is %g"%(global_step,accuracy_score))
                else:
                    print("No checkpoint file found")
                    return time.sleep((EVAL_INTERVAL_SEC))

def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
    evaluate(mnist)

if __name__=='__main__':
    tf.app.run()