# tf.contrib.learn usage

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000) # Q: batch_size? epochs?

estimator.fit(input_fn=input_fn, steps=1000) # Q: Where are the parameters to be esimated? Are they statically embedded into the estimator?

print(estimator.evaluate(input_fn=input_fn)) # Q: What if the input_fn is different to the fitted function?
