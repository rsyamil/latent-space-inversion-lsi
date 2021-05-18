# latent-space-inversion-lsi

## Prerequisites

The dataset used in this demo repository is the digit-MNIST images, **X**, with the forward model being a linear operator **G** and the resulting simulated responses denoted as **Y**. The physical system is represented as **Y=G(X)** or **D=G(M)**. More description on the dataset is available [here (in readme)](https://github.com/rsyamil/cnn-regression) and here is a [demo](https://github.com/rsyamil/dimensionality-reduction-autoencoders) on using convolutional autoencoders for dimensionality reduction of the digit-MNIST images.

![ForwardModel](/mnist/readme/forwardmodel.png)

We are interested in learning the inverse mapping **M=G'(D)** which is not trivial if the **M** is non-Gaussian (which is the case with the digit-MNIST images) and G is nonlinear (in this demo we assume a linear operator). Such complex mapping (also known as history-matching) may result in solutions that are non-unique with features that may not be consistent. LSI seeks to address those challenges and is an extension of [my early work](https://link.springer.com/article/10.1007/s10596-020-09971-4) that has more descriptions of the nature of this engineering problem.

## Implementation

LSI performs simultaneous dimensionality reduction (by extracting salient spatial features from **M** and temporal features from **D**) and inverse mapping (by mapping the salient features in **D** to **M**, i.e. latent spaces **z_d** and **z_m**). The architecture is composed of dual autoencoders connected with a regression model as shown below and is trained jointly. See [this Jupyter Notebook](https://github.com/rsyamil/latent-space-inversion-lsi/blob/main/mnist/qc-demo.ipynb) to train/load/QC the architecture.

![Arch](/mnist/readme/Archcombined.jpg)

The pseudocode is described here:

![Pseud](/mnist/readme/pseudocode.png)

## Demo of LSI Workflow

The trained LSI architecture is used to provide predictions of **m** for any given (and unseen) **d**. In [this demonstration](https://github.com/rsyamil/latent-space-inversion-lsi/blob/main/mnist/lsi-demo.ipynb), assume the following reference case of digit zero. The first left plot in the figure below represents the observed **d** and its reconstruction. The second plot shows the same, in scatter form. The third plot is the reference **m** and the fourth plot shows the prediction of LSI. 

![ref_all](/mnist/readme/test_sigs_ref_recons_demo.png)

Since the latent spaces **z_d** and **z_m** correspond to meaningful spatial and temporal variations in **D** and **M** respectively, they can be explored to obtain an ensemble of relevant inversion solutions. For example, the histograms below show **z_d** and the red line represents the latent variables **z_d** of the observed **d**. We can sample points around **z_d**, shown as the black unfilled bars denoted as "Pert".

![zds](/mnist/readme/test_zds_demo.png)

Similarly, the histograms below show **z_m** and the red line represents the  latent variables **z_m** of the reference **m**. The sampled points in the data latent space can be inversely mapped to the model latent space, shown as the black unfilled bars denoted as "Pert".

![zms](/mnist/readme/test_zms_demo.png)

The model decoder can be used to decode the sampled points around **z_m** to obtain the ensemble of relevant inversion solutions, as shown below. The inversion solutions show variations in which the digit zero can be written. 

![m_pert](/mnist/readme/test_m_pert_demo.png)

The ensemble of relevant inversion solutions is fed to the operator **G** (i.e. **D=G(M)**) to validate if the solutions can sufficiently reproduce the observed **d**. In the figure below, gray points/lines represent the entire testing dataset and the orange points/lines represent the simulated **D** from the ensemble, where a significant reduction in uncertainty is obtained.

![d_pert](/mnist/readme/test_d_pert_demo.png)

In practical applications, observed **d** can be noisy and LSI helps us to quickly obtain the ensemble of relevant inversion solutions that can be accepted within the noise level, as well as understand the variations of spatial features within the ensemble to improve the predictive power of the inversion solutions.
