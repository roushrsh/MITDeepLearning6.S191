import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import uuid
from tqdm import tqdm


import util.download_lung_data  #256 pixels

class PneumothoraxDataset:
    def __init__(self):
        print("Loading X-Ray Dataset!")

        train = h5py.File(util.download_lung_data.data_dir+'pneumothorax_train.h5','r')
        test = h5py.File(util.download_lung_data.data_dir+'pneumothorax_test.h5','r')

        self.X_train = train['image'][:]
        self.X_test = test['image'][:]
        self.Y_train = train['label'][:]
        self.Y_test = test['label'][:]

        self.num_train = self.X_train.shape[0]
        self.num_test = self.X_test.shape[0]

        self.batch_pointer = 0
        
    def getTotalNumDataPoints(self):
        return self.num_train+self.num_test
    
    def getTrainBatch(self, batch_size):
        inds = np.arange(self.batch_pointer,self.batch_pointer+batch_size)
        inds = np.mod( inds , self.num_train ) #cycle through dataset
        batch = (self.X_train[inds], self.Y_train[inds]) #grab batch

        self.batch_pointer += batch_size #increment counter before returning
        return batch

    def getTestBatch(self, batch_size):
        inds = np.random.choice(self.num_test, size=batch_size)
        return (self.X_test[inds], self.Y_test[inds])

data = PneumothoraxDataset()

print("Dataset consists of {} images".format(data.getTotalNumDataPoints()))



# TODO: Change value of INDEX here to visualize an image in the dataset!
#INDEX = 12

#image = data.X_train[INDEX]
#label = data.Y_train[INDEX]
#pred = np.argmax(label)

#plt.imshow(image[:,:,0], cmap='gray')
#print("This X-Ray "+("HAS" if pred else "DOES NOT have")+ " a pneumothorax")

from util.models import PneumothoraxDetectionModel
model = PneumothoraxDetectionModel()


batch_size = 15
learning_rate = 0.05
num_training_steps = int(1e6)

# Cost defines the empirical loss of the model given a set of labels and predictions. 
'''TODO: Fill in the cost function and arguments by looking at the model. Remember to keep track of logits!'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=model.y, logits=model.y_))

optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(cost)

correct_prediction = tf.equal(tf.argmax(model.y_, 1), tf.argmax(model.y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''TODO: Fill in the model prediction.'''
prediction = tf.argmax(model.y,1) # TODO

#tensorboard stuff
tf.summary.scalar('cost',cost) 
tf.summary.scalar('accuracy',accuracy)

merged_summary_op = tf.summary.merge_all() #combine into a single summary which we can run on Tensorboard
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

sess.run(init)
import uuid; #for tensorboard unique id
uniq_id = "./logs/lab2part2_"+uuid.uuid1().__str__()[:6]
summary_writer_train = tf.summary.FileWriter(uniq_id+'train', graph=tf.get_default_graph())
summary_writer_test = tf.summary.FileWriter(uniq_id+'test', graph=tf.get_default_graph())

#saver = tf.train.Saver()
#for step in range(num_training_steps):
#    (x_batch, y_batch) = data.getTrainBatch(batch_size) # get a training batch of data
#    _, trainLoss, summary = sess.run([optimizer, cost, merged_summary_op],
#                               feed_dict={model.x: x_batch, model.y:y_batch})

#    summary_writer_train.add_summary(summary, step) 

#    if step % 10 == 0:
#        (x_test, y_test) = data.getTestBatch(100) # get a testing batch of data
#        testLoss, testAcc, summary = sess.run([cost, accuracy, merged_summary_op], 
#                                              feed_dict={model.x: x_test, model.y:y_test})

#        print("step: {}, train: {}, \t\t test: {}, testAcc: {}".format(
#              step, trainLoss, testLoss, int(testAcc*100)))
#        summary_writer_test.add_summary(summary, step)

#    if step % 100 == 0:
#      save_path = saver.save(sess, uniq_id+'/model.ckpt')
#      print("Model saved in file: %s" % save_path)

saver = tf.train.Saver()
saver.restore(sess, "saved_model/model.ckpt")


from sklearn.metrics import roc_curve, auc


def compute_roc(y_true, y_score):
    """ Computes Receiving Operating Characteristic curve and area
    
    Params:
        - y_true: Ground truth binary labels
        - y_score: Continuous predictions in the range [0,1]
        
    Returns:
        - fpr: False Positive Rate at different thresholds
        - tpr: True Positive Rate at different thresholds
        - area: Area under the Receiving Operating Characteristic curve
    """
    '''TODO: Use the functions suggested above to fill in the ROC characteristics'''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auroc = auc(fpr, tpr)
    return fpr, tpr, auroc

def plot_roc(y_true, y_score, title='Receiver operating characteristic example'):
    """ Plots Receiving Operating Characteristic curve
    
    Params:
        - y_true: Ground truth binary labels
        - y_score: Continuous predictions in the range [0,1]
    """
    fpr, tpr, auroc = compute_roc(y_true, y_score)
    plt.figure(figsize=(10,6))
    plt.grid()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = {:.2f})'.format(auroc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend(loc="lower right", fontsize=14)
    plt.show()


NUMBER_TEST_SAMPLES = 10

y_true = []
y_score = []
for i in tqdm(range(NUMBER_TEST_SAMPLES)): #compute one at a time due to memory constraints
    y_true.extend( data.Y_test[[i],0] )
    probs = sess.run(model.probabilities, feed_dict={model.x: [data.X_test[i]]})
    y_score.extend( probs[:,0] )
    
correct = np.array(y_true) == np.round(y_score)
print("Accuracy = %2.2f%%" % (np.mean(correct)*100))


plot_roc(y_true, y_score, 'Receiver operating characteristic')



def extract_features_weights(sess, model):
    """ Extracts final feature maps and FC weights from trained model.
    
    Params:
        - sess: the current Tensorflow Session where the model was loaded
        - model: the PneumothoraxDetectionModel
    
    Returns (tuple):
        - an (_ x 16 x 16 x 512) tf.Tensor of activations from the final convolutional layer
        - an (512 x 2) tf.Tensor with the learned weights of the final FC layer
    """
    #access feature map activations directly from the model declaration
    feature_maps = model.skip4 
    
    #access the weights by searching by name
    dense_weights = sess.graph.get_tensor_by_name( os.path.split(model.y_.name)[0] + '/kernel:0')
    
    return (feature_maps, dense_weights)

(feature_maps, dense_weights) = extract_features_weights(sess, model)


print("Feature Maps: "+str(feature_maps))
print("Dense Weights: "+str(dense_weights))

def compute_cam(class_index, fmap, weights):
    """ Forms a CAM operation given a class name, feature maps, and weights
    
    Params: 
        - class_index: index of the class to measure
        - fmap: (1 x 16 x 16 x 512) tf.Tensor of activations from the final convolutional layer
        - weights: (512 x 2) tf.Tensor with the learned weights of the final FC layer
    
    Returns: 
        - (16 x 16) tf.Tensor of downscaled CAMs  
    """
    w_vec = tf.expand_dims(weights[:, class_index], 1) 
    
    _, h, w, c = fmap.shape.as_list()
    fmap = tf.squeeze(fmap) # remove batch dim
    fmap = tf.reshape(fmap, [h * w, c])
    
    '''TODO: compute the CAM! Remeber to look at the equation defining CAMs above to do this '''
    CAM = tf.matmul(fmap, w_vec) 
    CAM = tf.reshape(CAM, [1, h, w, 1]) 
    
    return CAM

cam = compute_cam(1, feature_maps, dense_weights) 

def upsample(cam, im_hw):
    """ Upsamples CAM to appropriate size
    
    Params:
        - cam: a x_x_ tf.Tensor
        - im_hw: target size in [H, W] format
        
    Returns:
        - Upsampled CAM with size _xHxW
    """
    '''TODO: upsampling function call. Hint: look at resize functions in tf.image'''
    upsampled = tf.image.resize_bilinear(cam, im_hw) 
    return upsampled


cam_upsampled = upsample(cam, [256,256]) 


def vis_cam(image, cam, save_file=None):
    """ Visualize class activation heatmap, overlaying on image.
    
    Params:
        - image: ndarray of size
    """
    cam = (cam - cam.min()) / (cam.max() - cam.min()) # TODO: check

    plt.imshow(255-image.squeeze(), cmap=plt.cm.gray, vmin=0, vmax=255) 
    plt.imshow(1-cam, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
    
    if save_file:
        plt.savefig(save_file)
    
    plt.show()
    plt.close()


# Plot some sample x-rays, predictions, and CAMs overlayed ontop
inds = [79,37,45,29,30]
for im, cl in zip(data.X_test[inds], data.Y_test[inds]):

    heatmap = sess.run(
        cam_upsampled,
        feed_dict={
            model.x: im[np.newaxis, :, :],
        })

    vis_cam(im, np.squeeze(heatmap))
