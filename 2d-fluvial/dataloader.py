import random
import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DataLoader:

	def __init__(self, simulator, verbose=False):

		self.verbose = verbose

		self.x_train = []
		self.x_test = []
		self.y_train = []
		self.y_test = []
		self.y_reg_train = []
		self.y_reg_test = []

		self.sim = simulator
		self.maxs = []

	def normalize(self, x):
		x_min = np.min(x)
		x_max = np.max(x)
		return (x - x_min)/(x_max - x_min)

	def load_data(self):
		'''create (simulate) a synthetic "time series" data vector (y) for each of the input (x) such that y=Gx and G is linear
		self.sim  represents some abstract function (i.e. fluid flow simulator)
		self.y_reg is presimulated
		'''
		x = np.load("data\M.npy")
		y_reg = np.load("data\D.npy")
		
		#reshape the models
		x_r = np.zeros([x.shape[0], 100, 100, 1])
		for i in range(x.shape[0]):
			x_r[i,:,:,:] = np.reshape(x[i,:], [1, 100, 100, 1])
		x = x_r
		
		self.maxs = np.max(y_reg, axis=0)
		y_reg = y_reg/self.maxs
		
		#create label, for every 500 models for 5 scenarios
		y = np.zeros([x.shape[0], ], dtype=np.int32)
		for i in range(5):
			y[i*500:i*500+500] = i
		
		#randomly sample from five scenarios
		np.random.seed(999)
		indexes = np.random.permutation(np.arange(0, x.shape[0], dtype=np.int32))
		partition = int(x.shape[0]*0.8)
		train_idx = indexes[0:partition]
		test_idx = indexes[partition:]

		self.x_train = x[train_idx]
		self.x_test = x[test_idx]
		self.y_train = y[train_idx]
		self.y_test = y[test_idx]
		self.y_reg_train = y_reg[train_idx]
		self.y_reg_test = y_reg[test_idx]

		if self.verbose: 
			print("Loaded training data x {:s} and y {:s} and y_labels {:s}".format(str(self.x_train.shape), str(self.y_reg_train.shape), str(self.y_train.shape)))
			print("Loaded testing data x {:s} and y {:s} and y_labels {:s}".format(str(self.x_test.shape), str(self.y_reg_test.shape), str(self.y_test.shape)))
		    
		return self.x_train, self.x_test, self.y_train, self.y_test, self.y_reg_train, self.y_reg_test

	def simulator(self, ms):
		'''simulate observations for a given set of models
		'''
		ms = np.where(ms<0.5, 0, 1)
		
		d_dim = self.sim.shape[-1]
		ds = np.zeros([ms.shape[0], d_dim])

		for i in range(ms.shape[0]):
			print("Running simulation ", i)
			ds[i:i+1, :] = np.reshape((ms[i:i+1, :, :, 0]), [1, ms.shape[1]*ms.shape[2]])@self.sim 
			ds[i:i+1, :] = ds[i:i+1, :] /np.squeeze(self.maxs)

		return np.expand_dims(ds, axis=-1)
             
 