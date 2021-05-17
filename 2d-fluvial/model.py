import numpy as np
import util
import keras
from keras.models import Model
from keras.layers import Layer, Flatten, LeakyReLU
from keras.layers import Input, Reshape, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from keras.layers import Conv1D, UpSampling1D
from keras.layers import AveragePooling1D, MaxPooling1D

from keras import backend as K
from keras.engine.base_layer import InputSpec

from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.losses import mse, binary_crossentropy
from keras import regularizers, activations, initializers, constraints
from keras.constraints import Constraint
from keras.callbacks import History, EarlyStopping

from keras.utils import plot_model, to_categorical
from keras.models import load_model

from keras.utils.generic_utils import get_custom_objects

import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style("whitegrid")

import matplotlib.cm as cm
from matplotlib.colors import Normalize

from sklearn.metrics import classification_report, confusion_matrix

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def RMSE(x, y):
    return np.sqrt(np.mean(np.square(x.flatten() - y.flatten())))

class GSI:

	def __init__(self, M, M_label, D, sample_ratio=0.3, simulator=None, class_names=[]):

		self.simulator = simulator

		self.M = M 
		self.M_label = M_label
		self.D = D  #pre-simulated
		self.class_names = class_names

		#for step 1 - classification 
		self.M_classify = []
		self.M_label_classify = []
		self.D_classify = []
		self.ratio = sample_ratio
		self.classifier = []
		self.proportion = []

		#for step 2 - inversion 
		self.regressor = []
		self.M_regression = []
		self.M_label_regression = []
		self.D_regression = []
        
	def collect_sampled_data_classification(self):
		'''collect data using sampled (at self.ratio) model realizations for step 1 - classification
		'''
		np.random.seed(999)
		indexes = np.random.choice(np.arange(0, self.M.shape[0]), size=int(self.M.shape[0]*self.ratio), replace=False)
		self.M_classify = self.M[indexes]
		self.M_label_classify = self.M_label[indexes]
		self.D_classify = self.D[indexes] #presimulated, otherwise, self.simulator(self.M_classify)
		print("Forward simulations ran : " + str(len(indexes)) + " out of " + str(self.M.shape[0]) + " models.")
        
	def encoder1D(self):

		input_dt = Input(shape=(self.D_classify.shape[1], self.D_classify.shape[2]))

		_ = Conv1D(64, 2, padding='same', data_format='channels_last')(input_dt)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = MaxPooling1D(2, padding="same")(_)

		_ = Conv1D(32, 4, padding='same', data_format='channels_last')(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = MaxPooling1D(2, padding="same")(_)

		_ = Conv1D(16, 6, padding='same', data_format='channels_last')(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = MaxPooling1D(2, padding="same")(_)

		_ = Flatten()(_)
		_ = Dense(32)(_)

		#encoded_d = Dense(2)(_)
		encoded_d = Dense(16)(_)
		out = Dense(5, activation = "softmax")(encoded_d)

		return input_dt, encoded_d, out
		
	def decoder2D(self, encoded_d):
	
	        #image decoder
		_ = Dense(13*13*32)(encoded_d)
		_ = Reshape((13, 13, 32))(_)

		_ = Conv2D(32, (5, 5), padding='same')(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = UpSampling2D((2, 2))(_)

		_ = Conv2D(16, (4, 4), padding='same')(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = UpSampling2D((2, 2))(_)

		_ = Conv2D(8, (3, 3))(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = UpSampling2D((2, 2))(_)

		decoded_m = Conv2D(1, (3, 3), padding='same')(_)

		return decoded_m
	
	def train_classifier(self, epoch=300, load=False):

		self.collect_sampled_data_classification()

		#classification model
		input_dt, encoded_d, out = self.encoder1D()
		self.classifier = Model(input_dt, out)
		self.classifier.compile(optimizer=Adam(lr=1e-3), loss="categorical_crossentropy", metrics=['accuracy'])
		self.classifier.summary()
		plot_model(self.classifier, to_file='readme/classifier.png')

		if not load:
			plot_losses = util.PlotLosses()
			self.classifier.fit(self.D_classify, to_categorical(self.M_label_classify),        
				epochs=epoch,
				batch_size=32,
				shuffle=True,
				validation_split=0.2,
				callbacks=[plot_losses])

			self.classifier.save('readme/classifier.h5')
		else:
			print("Trained model loaded")
			self.classifier = load_model('readme/classifier.h5')
            
	def inspect_classifier(self):
		'''test performance of classifier on a set of randomly sampled validation data
		'''
		np.random.seed(9999)
		indexes = np.random.choice(np.arange(0, self.M.shape[0]), size=int(self.M.shape[0]*self.ratio), replace=False)
		M_classify_val = self.M[indexes]
		M_label_classify_val = self.M_label[indexes]
		D_classify_val = self.D[indexes] #presimulated, otherwise, self.simulator(M_classify_val)
		
		M_label_classify_val_hat = self.classifier.predict(D_classify_val)

		matrix = confusion_matrix(M_label_classify_val, M_label_classify_val_hat.argmax(axis=1))
		f = util.plot_confusion_matrix(cm=matrix, classes=self.class_names, normalize=False, title='Confusion Matrix')
		f.savefig('readme/confusion_matrix.png')
		
		val_loss, val_acc = self.classifier.evaluate(D_classify_val,  to_categorical(M_label_classify_val), verbose=2)
		print('Validation accuracy:', val_acc)

	def get_proportion(self, d_obs, d_obs_label, m_ref):
		self.proportion = self.classifier.predict(d_obs)
		f = plt.figure(figsize=[8, 4])
		plt.subplot(1, 2, 1)
		util.plot_image(self.proportion, d_obs_label, m_ref, self.class_names)
		plt.subplot(1, 2, 2)
		util.plot_value_array(self.proportion, d_obs_label)
		plt.tight_layout()
		plt.show()
		f.savefig('readme/proportion.png')
	
	def collect_resampled_data_inversion(self):
		'''collect simulation data from a resampled set of relevant models
		  according to the proportion in self.proportion
		'''
		np.random.seed(997)
		proportion_count = np.squeeze((self.proportion*int(self.M.shape[0]*self.ratio)).astype(int))
		self.M_regression = np.zeros((np.sum(proportion_count), self.M.shape[1], self.M.shape[2], self.M.shape[3]))
		#self.D_regression = np.zeros((np.sum(proportion_count), self.D.shape[1], 1))
		cumm_count = 0
		for label, count in enumerate(proportion_count):
			#get "count" models with this label to use for inversion
			M_perlabel = self.M[self.M_label == label]
			#D_perlabel = self.D[self.M_label == label]
			indexes = np.random.choice(np.arange(0, M_perlabel.shape[0]), size=count, replace=False)
			self.M_regression[cumm_count:cumm_count+count] = M_perlabel[indexes]
			#self.D_regression[cumm_count:cumm_count+count] = D_perlabel[indexes]
			cumm_count += count
		self.D_regression = self.simulator(self.M_regression)
		print("Forward simulations ran : " + str(self.D_regression.shape[0]) + " out of " + str(self.M.shape[0]) + " models.")
		
	def train_regressor(self, epoch, load=False):
	
		self.collect_resampled_data_inversion()
		
		#regression model
		input_dt, encoded_d, out = self.encoder1D()
		decoded_m = self.decoder2D(encoded_d)
		
		self.regressor = Model(input_dt, decoded_m)
		self.regressor.compile(optimizer=Adam(lr=1e-3), loss="mean_squared_error", metrics=['mse'])
		self.regressor.summary()
		plot_model(self.regressor, to_file='readme/regressor.png')
		
		if not load:
			plot_losses = util.PlotLosses()
			self.regressor.fit(self.D_regression, self.M_regression,        
				epochs=epoch,
				batch_size=32,
				shuffle=True,
				validation_split=0.2,
				callbacks=[plot_losses])
			
			self.regressor.save('readme/regressor.h5')
		else:
			print("Trained model loaded")
			self.regressor = load_model('readme/regressor.h5')
			
	def inspect_regressor(self):
		'''test performance of regression model on a set of randomly sampled validation data
		from the resampled set "M_regression"
		'''
		np.random.seed(99)
		indexes = np.random.choice(np.arange(0, self.M_regression.shape[0]), size=int(self.M_regression.shape[0]*self.ratio), replace=False)
		M_regression_val = self.M_regression[indexes]
		D_regression_val = self.D_regression[indexes]
		
		M_regression_val_hat = self.regressor.predict(D_regression_val)
		
		#plot histogram
		bb = np.linspace(-0.05, 1.0, 50)
		fig = plt.figure(figsize=(5, 5))
		plt.hist(M_regression_val.flatten(), color='green', alpha=0.4, bins=bb, label="Val")
		plt.hist(M_regression_val_hat.flatten(), color='green', alpha=0.9, hatch='//', edgecolor='black', histtype='step', bins=bb, label="Pred")
		plt.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off', left='off', labelleft='off')
		plt.title('Validation RMSE_'+str(round(RMSE(M_regression_val, M_regression_val_hat),4)))
		plt.legend()
		fig.savefig('readme/hist.png')
		
		#show some samples, original vs predicted 
		num_rows = 2
		num_cols = 3
		num_images = num_rows*num_cols
		fig = plt.figure(figsize=(2*2*num_cols, 2*num_rows))
		for i in range(num_images):
			plt.subplot(num_rows, 2*num_cols, 2*i+1)
			plt.grid(False), plt.xticks([]), plt.yticks([])
			plt.imshow(M_regression_val[i], cmap='GnBu', vmin=0, vmax=1)
			if i < num_cols:
				plt.title('Reference')
			plt.subplot(num_rows, 2*num_cols, 2*i+2)
			plt.grid(False), plt.xticks([]), plt.yticks([])
			plt.imshow(M_regression_val_hat[i], cmap='GnBu', vmin=0, vmax=1)
			if i < num_cols:
				plt.title('Predicted.')
		plt.tight_layout()
		plt.show()
		fig.savefig('readme/comp.png')

	def get_inversion(self, d_obs, m_ref):
	
		#compare reference model "m_ref" with prediction given "d_obs"
		m_ref_hat = self.regressor.predict(d_obs)

		#run forward simulation on predicted model and compare with "d_obs"
		d_sim = self.simulator(m_ref_hat)
		
		fig = plt.figure(figsize=[10, 10], tight_layout=True)
		gs = gridspec.GridSpec(2, 2)
		
		ax = fig.add_subplot(gs[0, 0])
		ax.imshow(np.squeeze(m_ref), cmap='GnBu', vmin=np.min(m_ref), vmax=np.max(m_ref))
		plt.grid(False), plt.xticks([]), plt.yticks([])
		plt.title('Reference')
		
		ax = fig.add_subplot(gs[0, 1])
		ax.imshow(np.squeeze(m_ref_hat), cmap='GnBu', vmin=np.min(m_ref), vmax=np.max(m_ref))
		plt.grid(False), plt.xticks([]), plt.yticks([])
		plt.title('Inversion')
		
		ax = fig.add_subplot(gs[1, :])
		ax.plot(np.squeeze(d_obs), ls=':', c='k', label='True', alpha=0.9)
		ax.plot(np.squeeze(d_sim), c='r', label='Pred.', alpha=0.4)
		plt.ylim([np.min(d_obs)-0.15, np.max(d_obs)+0.15])
		plt.title('Data match RMSE_'+str(round(RMSE(d_obs, d_sim),4)))
		
		plt.legend()
		plt.show()
		fig.savefig('readme/inv.png')
		
		return m_ref_hat

		
		
		
		
		
	




		
		


        
        
        
        


    