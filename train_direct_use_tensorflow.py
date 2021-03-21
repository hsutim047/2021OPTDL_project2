import pdb
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import time
import math
import argparse

from net.net import CNN, CNN_model
from newton_cg import newton_cg
from utilities import read_data, predict, ConfigClass, normalize_and_reshape

def parse_args():
	parser = argparse.ArgumentParser(description='Newton method on DNN')
	parser.add_argument('--C', dest='C',
					  help='regularization term, or so-called weight decay where'+\
					  		'weight_decay = lr/(C*num_of_samples) in this implementation' ,
					  default=0.01, type=float)

	# Newton method arguments
	parser.add_argument('--GNsize', dest='GNsize',
					  help='number of samples for estimating Gauss-Newton matrix',
					  default=4096, type=int)
	parser.add_argument('--iter_max', dest='iter_max',
					  help='the maximal number of Newton iterations',
					  default=100, type=int)
	parser.add_argument('--xi', dest='xi',
					  help='the tolerance in the relative stopping condition for CG',
					  default=0.1, type=float)
	parser.add_argument('--drop', dest='drop',
					  help='the drop constants for the LM method',
					  default=2/3, type=float)
	parser.add_argument('--boost', dest='boost',
					  help='the boost constants for the LM method',
					  default=3/2, type=float)
	parser.add_argument('--eta', dest='eta',
					  help='the parameter for the line search stopping condition',
					  default=0.0001, type=float)
	parser.add_argument('--CGmax', dest='CGmax',
					  help='the maximal number of CG iterations',
					  default=250, type=int)
	parser.add_argument('--lambda', dest='_lambda',
					  help='the initial lambda for the LM method',
					  default=1, type=float)

	# SGD arguments
	parser.add_argument('--epoch_max', dest='epoch',
					  help='number of training epoch',
					  default=500, type=int)
	parser.add_argument('--lr', dest='lr',
					  help='learning rate',
					  default=0.01, type=float)
	parser.add_argument('--decay', dest='lr_decay',
					  help='learning rate decay over each mini-batch update',
					  default=0, type=float)
	parser.add_argument('--momentum', dest='momentum',
					  help='momentum of learning',
					  default=0, type=float)

	# Model training arguments
	parser.add_argument('--bsize', dest='bsize',
					  help='batch size to evaluate stochastic gradient, Gv, etc. Since the sampled data \
					  for computing Gauss-Newton matrix and etc. might not fit into memeory \
					  for one time, we will split the data into several segements and average\
					  over them.',
					  default=1024, type=int)
	parser.add_argument('--net', dest='net',
					  help='classifier type',
					  default='CNN_4layers', type=str)
	parser.add_argument('--train_set', dest='train_set',
					  help='provide the directory of .mat file for training',
					  default='data/mnist-demo.mat', type=str)
	parser.add_argument('--val_set', dest='val_set',
					  help='provide the directory of .mat file for validation',
					  default=None, type=str)
	parser.add_argument('--model', dest='model_file',
					  help='model saving address',
					  default='./saved_model/model.ckpt', type=str)
	parser.add_argument('--log', dest='log_file',
					  help='log saving directory',
					  default='./running_log/logger.log', type=str)
	parser.add_argument('--screen_log_only', dest='screen_log_only',
					  help='screen printing running log instead of storing it',
					  action='store_true')
	parser.add_argument('--optim', '-optim', 
					  help='which optimizer to use: SGD, Adam or NewtonCG',
					  default='NewtonCG', type=str)
	parser.add_argument('--loss', dest='loss', 
					  help='which loss function to use: MSELoss or CrossEntropy',
					  default='MSELoss', type=str)
	parser.add_argument('--dim', dest='dim', nargs='+', help='input dimension of data,'+\
						'shape must be:  height width num_channels',
					  default=[32, 32, 3], type=int)
	parser.add_argument('--seed', dest='seed', help='a nonnegative integer for reproducibility',
					  default=0, type=int)	  
	args = parser.parse_args()
	return args

args = parse_args()

def init_model(param):
	init_ops = []
	for p in param:
		if 'kernel' in p.name:
			weight = np.random.standard_normal(p.shape)* np.sqrt(2.0 / ((np.prod(p.get_shape().as_list()[:-1]))))
			opt = tf.compat.v1.assign(p, weight)
		elif 'bias' in p.name:
			zeros = np.zeros(p.shape)
			opt = tf.compat.v1.assign(p, zeros)
		init_ops.append(opt)
	return tf.group(*init_ops)

def gradient_trainer(config, sess, network, full_batch, val_batch, saver, test_network):

	x, y = full_batch[0], full_batch[1]

	global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
	learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[], name='learning_rate')
	
	model = CNN_model(config.net, config.dim, config.num_cls)

	reg = 0.0

	param = model.trainable_weights

	wfile = open("direct_tensorflow.txt", "w+")
	sess.run(init_model(param))
	weights = [sess.run(v) for v in param]
	print(weights, file = wfile)
	print(weights[0][0][0][0])
	for p in param:
		reg = reg + tf.reduce_sum(input_tensor=tf.pow(p,2))
	reg_const = 1/(2*config.C)

	loss_with_reg = lambda y_true, y_pred: reg_const*reg + tf.reduce_mean(tf.reduce_sum(
 					tf.square(y_true - y_pred), axis=1))

	opt = tf.compat.v1.train.MomentumOptimizer(learning_rate = config.lr,
		  momentum = config.momentum)

	#print(config.args)

	model.compile(optimizer = opt, loss = loss_with_reg)

	for i in range(len(full_batch[0]) // config.bsize):
		model.fit(x[i * config.bsize: (i + 1) * config.bsize],
				  y[i * config.bsize: (i + 1) * config.bsize],
				  batch_size = config.bsize, verbose = 0)
		weights = [sess.run(v) for v in param]
		print(weights, file = wfile)
	"""
	model.fit(x, y, batch_size = config.bsize, shuffle = True)
	weights = [sess.run(v) for v in param]
	print(weights[0][0][0][0], file = wfile)
	"""
	print(weights[0][0][0][0])

	wfile.close()

def newton_trainer(config, sess, network, full_batch, val_batch, saver, test_network):

	_, _, loss, outputs = network
	newton_solver = newton_cg(config, sess, outputs, loss)
	sess.run(tf.compat.v1.global_variables_initializer())

	print('-------------- initializing network by methods in He et al. (2015) --------------')
	param = tf.compat.v1.trainable_variables()
	sess.run(init_model(param))
	newton_solver.newton(full_batch, val_batch, saver, network, test_network)


def main():

	full_batch, num_cls, label_enum = read_data(filename=args.train_set, dim=args.dim)
	
	if args.val_set is None:
		print('No validation set is provided. Will output model at the last iteration.')
		val_batch = None
	else:
		val_batch, _, _ = read_data(filename=args.val_set, dim=args.dim, label_enum=label_enum)

	num_data = full_batch[0].shape[0]
	
	config = ConfigClass(args, num_data, num_cls)

	if isinstance(config.seed, int):
		tf.compat.v1.random.set_random_seed(config.seed)
		np.random.seed(config.seed)

	if config.net in ('CNN_4layers', 'CNN_7layers', 'VGG11', 'VGG13', 'VGG16','VGG19'):
		x, y, outputs = CNN(config.net, num_cls, config.dim)
		test_network = None
	else:
		raise ValueError('Unrecognized training model')

	if config.loss == 'MSELoss':
		loss = tf.reduce_sum(input_tensor=tf.pow(outputs-y, 2))
	else:
		loss = tf.reduce_sum(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
	
	network = (x, y, loss, outputs)

	sess_config = tf.compat.v1.ConfigProto()
	sess_config.gpu_options.allow_growth = True

	with tf.compat.v1.Session(config=sess_config) as sess:
		
		full_batch[0], mean_tr = normalize_and_reshape(full_batch[0], dim=config.dim, mean_tr=None)
		if val_batch is not None:
			val_batch[0], _ = normalize_and_reshape(val_batch[0], dim=config.dim, mean_tr=mean_tr)

		param = tf.compat.v1.trainable_variables()

		mean_param = tf.compat.v1.get_variable(name='mean_tr', initializer=mean_tr, trainable=False, 
					validate_shape=True, use_resource=False)
		label_enum_var=tf.compat.v1.get_variable(name='label_enum', initializer=label_enum, trainable=False,
					validate_shape=True, use_resource=False)
		saver = tf.compat.v1.train.Saver(var_list=param+[mean_param])
		
		if config.optim in ('SGD', 'Adam'):
			gradient_trainer(
				config, sess, network, full_batch, val_batch, saver, test_network)
		elif config.optim == 'NewtonCG':
			newton_trainer(
				config, sess, network, full_batch, val_batch, saver, test_network=test_network)


if __name__ == '__main__':
	main()

