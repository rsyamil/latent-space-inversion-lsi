import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np

import keras
from IPython.display import clear_output
import matplotlib as mpl
import itertools

#plot function for sample images
def plot_tile(samples):
    
	num_samples, x_dim, y_dim, _ = samples.shape
	axes = (np.round(np.sqrt(num_samples))).astype(int)
	fig = plt.figure(figsize=(axes, axes))
	gs = gridspec.GridSpec(axes, axes)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_aspect('equal')
		plt.imshow(sample, cmap=plt.get_cmap('viridis'), aspect='auto')
        
#visualize the generated signals (for training dataset) 
def plot_signals(y_reg_train, labels):

	fig, ax = plt.subplots(1,1, figsize = (16, 7))

	my_cmap = cm.get_cmap('jet')
	my_norm = Normalize(vmin=0, vmax=9)
	cs = my_cmap(my_norm(labels))

	for j in range(10):
		plt.subplot(2, 5, j+1)
		for i in range(500):
			if (labels[i] == j):
				plt.plot(y_reg_train[i, :], c=cs[i], alpha=0.5)
		plt.ylim([0, 1])
		plt.title('digit '+str(j))
        
	return fig

#function to view training and validation losses
class PlotLosses(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.i = 0
		self.x = []
		self.losses = []
		self.val_losses = []
		self.fig = plt.figure()
		self.logs = []

	def on_epoch_end(self, epoch, logs={}):
		self.logs.append(logs)
		self.x.append(self.i)
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.i += 1
		clear_output(wait=True)
		plt.plot(self.x, self.losses, label="loss", c = 'green')
		plt.plot(self.x, self.val_losses, label="val_loss", c = 'red')
		plt.legend()
		plt.show()
        
#function to view multiple losses
def plotAllLosses(loss1, loss2):         
	N, m1f = loss1.shape
	_, m2f = loss2.shape

	print(loss1.shape)
	print(loss2.shape)

	fig = plt.figure(figsize=(6, 12))
	plt.subplot(2, 1, 1)
	plt.plot(loss1[:, 0], label='loss1_check1')
	plt.plot(loss1[:, 1], label='loss1_check2')
	plt.plot(loss1[:, 2], label='loss1_check3')
	plt.plot(loss1[:, 3], label='loss1_check4')
	plt.plot(loss1[:, 4], label='loss1_check3')
	plt.plot(loss1[:, 5], label='loss1_check4')
	plt.legend()

	plt.subplot(2, 1, 2)
	plt.plot(loss2[:, 0], label='loss2_check1')
	plt.plot(loss2[:, 1], label='loss2_check2')
	plt.legend()

	return fig
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.jet):

	fig = plt.figure(figsize=[8, 8])
	ax = plt.axes()
	plt.imshow(cm, interpolation='nearest', cmap=cmap, alpha=0.7)
	plt.title(title)

	plt.colorbar(fraction=0.046, pad=0.04) 

	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	#print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment="center", color="black" if cm[i, j] > thresh else "black")

	plt.grid(False)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	return fig

def plot_image(proportion, d_obs_label, m_ref, class_names):
	true_label, img = d_obs_label, np.squeeze(m_ref)
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap=plt.cm.jet)
	predicted_label = np.argmax(proportion)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'
	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
				100*np.max(proportion),
				class_names[int(true_label)]),
				color=color)

def plot_value_array(proportion, d_obs_label):
	true_label = np.squeeze(d_obs_label)
	plt.grid(False)
	
	plt.xticks(range(5))
	plt.yticks([])
	thisplot = plt.bar(range(5), np.squeeze(proportion), color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(proportion)
	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

def normalize(arr):
	arr_min = np.min(arr)
	return (arr-arr_min)/(np.max(arr)-arr_min)

def plot_cube(cube, angle=320):
	'''cube must be 3d
	'''
	cube = normalize(cube)
	cube = np.expand_dims(cube, axis=-1)            #28x28x1
	facecolors = cm.jet(cube)                      #28x28x4
	filled = np.ones(cube.shape)                    #28x28x1

	x, y, z = np.indices(np.array(cube.shape)+1)    #29x29x2 for each axis

	fig = plt.figure(figsize=[6, 6])  
	ax = fig.gca(projection='3d')
	ax.view_init(30, angle)
	ax.set_axis_off()
	ax.set_box_aspect((cube.shape[0], cube.shape[1], 20))

	ax.voxels(x, y, z, filled = filled, facecolors=facecolors, linewidth=0.0001)

	plt.show()

    
    
    

				