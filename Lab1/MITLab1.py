import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#a =tf.constant(15, name = "a")
#b = tf.constant(61, name ="b")

#c= tf.add(a,b, name="c")
#print(c)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = tf.add(a,b, name = "c")
d = tf.subtract(b,1, name = "d")
e = tf.multiply(c,d, name = "e")

with tf.Session() as session:
	a_data, b_data = 2.0, 4.0
	#data
	feed_dict = {a: a_data, b: b_data}
	#pass data and run graph
	output = session.run([e], feed_dict=feed_dict)
	print(output)
