import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n_input_nodes =2
n_output_nodes = 1
x = tf.placeholder(tf.float32, (None, n_input_nodes))
w = tf.Variable(tf.ones((n_input_nodes, n_output_nodes)), dtype=tf.float32)
b = tf.Variable(tf.zeros(n_output_nodes), dtype=tf.float32)


#z = tf.add(x*w, b, name = "z")
#temp = tf.exp(-z, name = "temp") #tf.math.exp(-x, name = "temp")
z = tf.add(tf.matmul(x,w, name = "temp"), b, name = "z")

#out = tf.divide(1.0, 1.0+temp, name="out")
out = tf.sigmoid(z, name = "out")

test_input = [[0.25, 0.15]]
graph = tf.Graph()

with tf.Session() as session:
	tf.global_variables_initializer().run(session=session)
	feed_dict = {x: test_input}
	output = session.run([out], feed_dict=feed_dict)
#	print z
	print output[0]
