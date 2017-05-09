# Note. This code belongs to the tensorflow tutorial. 
# Note. This version is about to find the optimal LINEAR parameters.
# i.e., y = xW* + b*, 
# where x: a vectorized image(1-by-784), 
# y: the corresponding label(10-by-1), 
# W*(784-by-10) and b*(10-by-1) are the trained parameters. 

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
# y = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

ss = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    ss.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(ss.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
