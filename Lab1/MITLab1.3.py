import os

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
import tensorflow as tf


def f(x):
    # f(x) = x^2 + 3
    return tf.multiply(x, x) + 3

print( "f(4) = %.2f" % f(4.) )

# First order derivative
df = tfe.gradients_function(f) # tfe == eager mode
print( "df(4) = %.2f" % df(4.)[0] )

# Second order derivative
'''TODO: fill in the expression for the second order derivative using Eager mode gradients'''
d2f = tfe.gradients_function(df)
print( "d2f(4) = %.2f" % d2f(4.)[0] )



a = tf.constant(12)
counter = 0
while not tf.equal(a, 1):
  if tf.equal(a % 2, 0):
    a = a / 2
  else:
    a = 3 * a + 1
  print(a)
