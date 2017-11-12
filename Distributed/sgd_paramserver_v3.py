'''
Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python example.py --job_name="ps" --task_index=0 
pc-02$ python example.py --job_name="worker" --task_index=0 
pc-03$ python example.py --job_name="worker" --task_index=1 
pc-04$ python example.py --job_name="worker" --task_index=2 

More details here: ischlag.github.io
'''

from __future__ import print_function

import tensorflow as tf
import sys
import time

# cluster specification
parameter_servers = ["10.24.1.207:3222"]
workers = [	"10.24.1.213:3232", 
			"10.24.1.217:2321",
			"10.24.1.219:3222"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)

# config
batch_size = 100
learning_rate = 0.0005
training_epochs = 20
logs_path = "/tmp/sgd/"

# loading data set ######################################################



from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder


Nfeatures = 2000
f =  open('verysmall_train.txt', 'r') 
cls = set()
text_samples = []
label_samples = []
for article in f:
    try :        
        labels, line = article.split('\t')
    except :
        continue
    labels = labels.split(',')
    text_samples.append(line)
    labels = [label.replace(" ","") for label in labels]
    label_samples.append(labels)
    for label in labels:
        cls.add(label)
f.close()

trans = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, max_features=Nfeatures)
features = trans.fit_transform(text_samples)
N_class = len(cls)
y= np.zeros((N_class))

encoder = {}
for i, l in enumerate(list(cls)):
    encoder[l] = i 

one_hot_encoded_label = []
for label_i in label_samples:
    y = np.zeros((N_class))
    for lab in label_i :
        y[encoder[lab]] = 1
        
    one_hot_encoded_label.append(y)
        
lbda = 0.1
mu = 0.01


f =  open('verysmall_test.txt', 'r') 

text_test = []
label_test = []
for article in f:
    try :        
        labels, line = article.split('\t')
    except :
        continue
    labels = labels.split(',')
    text_test.append(line)
    labels = [label.replace(" ","") for label in labels]
    label_test.append(labels)
    
f.close()

feature_test = trans.transform(text_test)
one_hot_encoded_label_test = []
for label_i in label_test:
    y_encode = np.zeros((N_class))
    for lab in label_i :
        y_encode[encoder[lab]] = 1
        
    one_hot_encoded_label_test.append(y_encode)
    





################################################################################

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

	# Between-graph replication
	with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):

		# count the number of updates
		global_step = tf.get_variable(
            'global_step',
            [],
            initializer = tf.constant_initializer(0),
			trainable = False)

		# input images
		with tf.name_scope('input'):
		  # None -> batch size can be any size, 784 -> flattened mnist image

		  x = tf.placeholder(tf.float32, [None, Nfeatures])

		  y_ = tf.placeholder(tf.float32, [None, N_class])

		# model parameters will change during training so we use tf.Variable
		tf.set_random_seed(1)
		with tf.name_scope("weights"):
			W = tf.Variable(tf.zeros((Nfeatures, N_class )), dtype=tf.float32)
			

		# implement model
		with tf.name_scope("softmax"):
			# y is our prediction
			y = 1/ (1 + tf.exp( -1 * tf.matmul(x, W) ) )

		# specify cost function
		with tf.name_scope('cross_entropy'):
			# this is our cost
			cross_entropy = tf.reduce_mean(
                -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

		# specify optimizer
		with tf.name_scope('train'):
			# optimizer is an "operation" which we can execute in a session
			error =  y_ - y
			update = tf.matmul(tf.transpose(x) ,  error) -2* mu *W 
			train_op = W.assign_add(lbda * update) 


		with tf.name_scope('Accuracy'):
			# accuracy
			correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                with tf.name_scope('Prediction'):
                        prediction = tf.argmax(y, 1)

		# create a summary for our cost and accuracy
		tf.summary.scalar("cost", cross_entropy)
		tf.summary.scalar("accuracy", accuracy)

		# merge all summaries into a single "operation" which we can execute in a session 
		summary_op = tf.summary.merge_all()
		init_op = tf.global_variables_initializer()
		print("Variables initialized ...")

	sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
														global_step=global_step,
														init_op=init_op)

	begin_time = time.time()
	frequency = 100
	with sv.prepare_or_wait_for_session(server.target) as sess:
		'''
		# is chief
		if FLAGS.task_index == 0:
			sv.start_queue_runners(sess, [chief_queue_runner])
			sess.run(init_token_op)
		'''
		# create log writer object (this will log on every machine)
		writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
				
		# perform training cycles
		start_time = time.time()
		training_epochs = 30
		for epoch in range(training_epochs):

			# number of batches in one epoch
			#batch_count = int(mnist.train.num_examples/batch_size)
                        #loss = 0
			#for i in range(features.shape[0]):
			loss =  sess.run(cross_entropy, feed_dict={x: features.toarray(), y_: one_hot_encoded_label})
			print((epoch, loss)) 
			#count = 0
			for i in range(features.shape[0]):
				#batch_x, batch_y = mnist.train.next_batch(batch_size)
				
				#sess.run(assig_op, cross_entropy],feed_dict={x: features[i].toarray(), y: [one_hot_encoded_label[i]]})    

				# perform the operations we defined earlier on batch
				sess.run(train_op, feed_dict={x: features[i].toarray(), y_: [one_hot_encoded_label[i]]})
		#loss = 0
		#for i in range(features.shape[0]):
		loss =   sess.run(cross_entropy,feed_dict={x: features[i].toarray(), y_ : one_hot_encoded_label})
		print((epoch+1, loss))				

        
        	acc = 0
		print("Training time")
		print(time.time()-start_time)
        	print("Training performance")
        	for i in range(features.shape[0]):
            		y_predict = sess.run(prediction, feed_dict={x: features[i].toarray(), y: [one_hot_encoded_label[i]]} )
            		if one_hot_encoded_label[i][y_predict[0]] == 1 :
                        	acc = acc + 1
		print((acc, i))
                
        	print(acc*1./(features.shape[0]))
        	testing_time = time.time()
		print("Testing performance")
        	acc = 0
        	for i in range(feature_test.shape[0]):
            		y_predict = sess.run(prediction, feed_dict={x: feature_test[i].toarray(), y: [one_hot_encoded_label_test[i]]} )
            		if one_hot_encoded_label_test[i][y_predict[0]] == 1 :
                		acc = acc + 1
            	print((acc, i))
        	print(acc*1./(feature_test.shape[0]) )
		print("Testing time")
		print(time.time()-testing_time)
	sv.stop()
	print("done")
