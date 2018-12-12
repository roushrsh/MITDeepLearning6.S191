import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from util.util import print_progress
from util.create_dataset import create_dataset, get_batch
from util.midi_manipulation import noteStateMatrixToMidi

min_song_length  = 128
encoded_songs    = create_dataset(min_song_length)

NUM_SONGS = len(encoded_songs)
#print(str(NUM_SONGS) + " total songs to learn from")
#print(encoded_songs[0].shape)

input_size       = encoded_songs[0].shape[1]   # The number of possible MIDI Notes
output_size      = input_size                  # Same as input size
hidden_size      = 128                         # Number of neurons in hidden layer

learning_rate    = 0.001 # Learning rate of the model
training_steps   = 200  # Number of batches during training
batch_size       = 64    # Number of songs per batch
timesteps        = 64    # Length of song snippet -- this is what is fed into the model

assert timesteps < min_song_length


input_placeholder_shape = [None, timesteps, input_size] #TODO
output_placeholder_shape = [None, output_size] #TODO

input_vec  = tf.placeholder("float", input_placeholder_shape)  
output_vec = tf.placeholder("float", output_placeholder_shape)  

# Define weights
weights = tf.Variable(tf.random_normal([hidden_size, output_size])) 

'''TODO: Define biases by sampling from a Normal distribution.'''
biases = tf.Variable(tf.random_normal([output_size]))


def RNN(input_vec, weights, biases):
    input_vec = tf.unstack(input_vec, timesteps, 1)

    lstm_cell = rnn.BasicLSTMCell(hidden_size) 

    outputs, states = rnn.static_rnn(lstm_cell, input_vec, dtype=tf.float32)

    # Recall that this is just a linear operation: xW + b, where W is the set of weights and b the biases.
    recurrent_net = tf.matmul(outputs[-1], weights) + biases # TODO 
    
    '''TODO: Use the TensorFlow softmax function to output a probability distribution over possible notes.'''
    prediction = tf.nn.softmax(recurrent_net) # TODO
    
    # All that's left is to return recurrent_net (the RNN output) and prediction (the softmax output)
    return recurrent_net, prediction


logits, prediction = RNN(input_vec, weights, biases)


#loss function using softmax
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=output_vec))

#optimizer, using learning rate we had prev
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

#Train using optimizer
train_op = optimizer.minimize(loss_op)

#see if true note hits predicted
true_note = tf.argmax(output_vec,1)
pred_note = tf.argmax(prediction, 1) 
correct_pred = tf.equal(pred_note, true_note)

#What is our accuracy?
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#initializer variables
init = tf.global_variables_initializer()

###Let's start now

#launch sessions
sess = tf.InteractiveSession()

#Init variables
sess.run(init)

#Train
display_step = 100 #How often to show what's happening

for step in range(training_steps):

	batch_x, batch_y = get_batch(encoded_songs, batch_size, timesteps, input_size, output_size)

	feed_dict = {
        	input_vec: batch_x, 
        	output_vec: batch_y 
        	}
	sess.run(train_op, feed_dict=feed_dict)

	
 	if step % display_step == 0 or step == 1:
        	# LOSS, ACCURACY: Compute the loss and accuracy by running both operations 
        	loss, acc = sess.run([loss_op, accuracy_op], feed_dict=feed_dict)     
        	suffix = "\nStep " + str(step) + ", Minibatch Loss= " + \
        	         "{:.4f}".format(loss) + ", Training Accuracy= " + \
        	         "{:.3f}".format(acc)

        	print_progress(step, training_steps, barLength=50, suffix=suffix)



import matplotlib.pyplot as plt
GEN_SEED_RANDOMLY = 222 # for generating new songs
if GEN_SEED_RANDOMLY:
    ind = np.random.randint(NUM_SONGS)
else:
    ind = 41 # "How Deep is Your Love" by Calvin Harris as a starting seed
 
gen_song = encoded_songs[ind][:timesteps].tolist() 

for i in range(500):
    seed = np.array([gen_song[-timesteps:]])
    # Use our RNN for prediction using our seed! 

	#Using our predicted, not training
    predict_probs = sess.run(prediction, feed_dict = {input_vec:seed}) # TODO

    # Define output vector for our generated song by sampling from our predicted probability distribution
    played_notes = np.zeros(output_size)
  #  print np.argmax(predict_probs[0])
    plt.plot(predict_probs[0])
    sampled_note = np.random.choice(range(output_size), p=predict_probs[0]) # TODO
    #sampled_note = np.argmax(predict_probs[0])
    played_notes[sampled_note] = 1
    gen_song.append(played_notes)

noteStateMatrixToMidi(gen_song, name="generated/gen_song_0")
noteStateMatrixToMidi(encoded_songs[ind], name="generated/base_song_0")
print("saved generated song! seed ind: {}".format(ind))

