import sys
import random
import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import dataloader
import util
import autoencoderdual

if __name__ == "__main__":

    #get input from command line
    load = True
    load = sys.argv[1]
    
    #load data
    dataset = dataloader.DataLoader(verbose=True)
    x_train, x_test, y_train, y_test, y_reg_train, y_reg_test = dataset.load_data()

    #visualize first 9 samples of input images/data based on class labels, within the training dataset
    unique_labels, label_counts = np.unique(y_train, return_counts=True)
    for label in unique_labels:
        x_train_perlabel = x_train[np.squeeze(y_train) == label]
        fig = util.plot_tile(x_train_perlabel[0:9, :, :])
        
    fig = util.plot_signals(y_reg_train, y_train)
    fig.savefig('readme/signals.png')

    #load trained architecture, to retrain set "load=False", ~1hr on NVIDIA RTX2080Ti
    LSI = autoencoderdual.Autoencoder(x_train, y_reg_train)
    LSI.train_autoencoder_dual(epoch=200, load=load)

    autoencoderdual.inspect_LSI(LSI, x_test, y_reg_test, y_test)
    autoencoderdual.inspect_LSI_z(LSI, x_test, y_reg_test, y_test)
    
    print("QC complete")
    