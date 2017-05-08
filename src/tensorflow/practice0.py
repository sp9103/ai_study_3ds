import os
# below statement is about to ignore SEE warning. 
# another way: export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


##############################################################
## Basic operation 1

3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
#output: 
# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)

ss = tf.Session() # Q: what is the difference between upper and lower camel cases?
print(ss.run([node1, node2]))
#output:
# [3.0, 4.0]

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("ss.run(node3): ", ss.run(node3))
#output:
# node3:  Tensor("Add_2:0", shape=(), dtype=float32)
# ss.run(node3):  7.0


##############################################################
## Basic operation 2

a = tf.placeholder(tf.float32) # placeholder: external input
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)
print(ss.run(adder_node, {a: 3, b: 4.5}))
#output: 7.5
print(ss.run(adder_node, {a: [1,3], b: [2, 4]}))
#output: [ 3.  7.]

add_and_triple = adder_node * 3.
print(ss.run(add_and_triple, {a: 3, b: 4.5})) 
#output: 22.5

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
ss.run(init)
print(ss.run(linear_model, {x: [1, 2, 3, 4]}))
#output: [ 0.          0.30000001  0.60000002  0.90000004]

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(ss.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
#output: 23.66

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
ss.run([fixW, fixb])
print(ss.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
#output: 0.0


##############################################################
## tf.train API

optimizer = tf.train.GradientDescentOptimizer(0.01)
# squared_deltas = tf.square((W*x+b) - y)
# loss = tf.reduce_sum(squared_deltas)
train = optimizer.minimize(loss)

# init = tf.global_variables_initializer()
ss.run(init) # reset values to incorrect defaults.
for i in range(1000):
    ss.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(ss.run([W, b]))
#output: [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]

