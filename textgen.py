import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf, numpy as np

def onehot(w,v):
	'''Generate one-hot rep'''
	out = [0]*len(v)
	out[v.index(w)] =1
	return out

if __name__ == '__main__':
	# parameters
	v = '#abcdefghijklmnopqrstuvwxyz' # vocabulary
	nx = len(v) #num of poss inputs
	ny = len(v) #num of poss outputs
	nt = 4 #num of time steps
	ls = 10 #lstm-size (num of units)

	#Build graph
	input_batch = tf.placeholder(tf.float32, [None, nt, nx])
	Wy = tf.Variable(tf.random_normal([ls,ny]))
	by = tf.Variable(tf.random_normal([ny]))
	x = tf.unstack(input_batch, nt, 1)
	lstm = tf.contrib.rnn.BasicLSTMCell(ls)
	outputs, state = tf.contrib.rnn.static_rnn(lstm, x, dtype=tf.float32)
	y = tf.matmul(outputs[-1], Wy) + by
	y = tf.nn.softmax(y)
	
	#Run Session
	saver = tf.train.Saver()
	sess = tf.Session()
	saver.restore(sess, './model.ckpt')
	out = ''
	bos = []
	for i in range(nt): bos.append(onehot('#', v))
	history = bos
	pc = None
	while pc != '#':
		prb = sess.run(y, feed_dict={input_batch:[history]})
		vec = np.random.multinomial(1, prb[0])
		pc = v[vec.argmax()]
		out += pc
		history = history[1:]
		history.append(vec)
	print out
	sess.close()

			
