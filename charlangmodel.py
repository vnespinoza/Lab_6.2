import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf, numpy as np


def onehot(w,v):
	'''Generate one-hot rep'''
	out = [0]*len(v)
	out[v.index(w)] =1
	return out

def gen_data(nt,v,file):
	'''Load words in file into sequence data'''
	inputs = []; labels = []
	f = open(file)
	for line in f:
		w ='#'*nt+line.strip()+'#'
		for i in range(len(w)-nt):
			x = []
			for c in w[i:i+nt]: x.append(onehot(c,v))
			inputs.append(x)
			y = onehot(w[i+nt], v)
			labels.append(y)
	f.close()
	return inputs,labels

if __name__ == '__main__':
	# parameters
	v = '#abcdefghijklmnopqrstuvwxyz' # vocabulary
	nx = len(v) #num of poss inputs
	ny = len(v) #num of poss outputs
	nt = 4 #num of time steps
	ls = 10 #lstm-size (num of units)
	bs = 10 #batch size
	ne = 100 # num of epochs
	
	#Generate data
	inputs, labels = gen_data(nt, v, './words.txt')
	#Build graph
	input_batch = tf.placeholder(tf.float32, [None, nt, nx])
	label_batch = tf.placeholder(tf.float32,[None, ny])
	Wy = tf.Variable(tf.random_normal([ls,ny]))
	by = tf.Variable(tf.random_normal([ny]))
	lstm = tf.contrib.rnn.BasicLSTMCell(ls)
	x = tf.unstack(input_batch, nt, 1)
	outputs, state = tf.contrib.rnn.static_rnn(lstm, x, dtype=tf.float32)
	y = tf.matmul(outputs[-1], Wy) + by
	
	#Define training step
	loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label_batch,y))
	train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	saver = tf.train.Saver()

	#Run Session
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for i in range(ne):
		sys.stderr.write('# Epoch '+str(i+1)+'...\r')
		for j in range(len(inputs)/bs):
			ib = inputs[j*bs:(j+1)*bs]
			lb = labels[j*bs:(j+1)*bs]
			sess.run(train, feed_dict={input_batch:ib, label_batch:lb})
	sys.stderr.write('#Training complete\n')
	saver.save(sess,'./model.ckpt')
	sys.stderr.write('#Model saved in ./model.ckpt\n')
	sess.close()

			
