# latent-space-inversion-lsi

This repository is to supplement [**H052-08 - Latent-Space Inversion (LSI) for Subsurface Flow Model Calibration with Physics-Informed Autoencoding**](https://agu.confex.com/agu/fm20/meetingapp.cgi/Paper/753910) as presented at the American Geophysical Union Fall Meeting 2020 (AGUFM2020) and [a paper currently in review](https://scholar.google.com/citations?user=mQUFzL8AAAAJ&hl=en#d=gs_md_cita-d&u=%2Fcitations%3Fview_op%3Dview_citation%26hl%3Den%26user%3DmQUFzL8AAAAJ%26citation_for_view%3DmQUFzL8AAAAJ%3AYsMSGLbcyi4C%26tzom%3D480). The AGUFM2020 [video](https://vimeo.com/495980342) presentation can be viewed below:

[![Video](/readme/thumb2.png)](https://vimeo.com/495980342)

## Prerequisites

The dataset used in this demo repository is the digit-MNIST images, **X**, with the forward model being a linear operator **G** and the resulting simulated responses denoted as **Y**. The physical system is represented as **Y=G(X)** or **D=G(M)**. More description on the dataset is available [here (in readme)](https://github.com/rsyamil/cnn-regression) and here is a [demo](https://github.com/rsyamil/dimensionality-reduction-autoencoders) on using convolutional autoencoders for dimensionality reduction of the digit-MNIST images.

![ForwardModel](/readme/forwardmodel.png)

We are interested in learning the inverse mapping **M=G'(D)** which is not trivial if the **M** is non-Gaussian (which is the case with the digit-MNIST images) and G is nonlinear (in this demo we assume a linear operator). Such complex mapping (also known as history-matching) may result in solutions that are non-unique with features that may not be consistent. LSI seeks to address those challenges and is an extension of [my early work](https://link.springer.com/article/10.1007/s10596-020-09971-4) that has more descriptions of the nature of this engineering problem.

## Implementation

LSI performs simultaneous dimensionality reduction (by extracting salient spatial features from **M** and temporal features from **D**) and inverse mapping (by mapping the salient features in **D** to **M**, i.e. latent spaces). The architecture is composed of dual autoencoders connected with a regression model as shown below and is trained jointly. See [this Jupyter Notebook](https://github.com/rsyamil/latent-space-inversion-lsi/blob/main/qc-demo.ipynb) to train/load/QC the architecture.

![Arch](/readme/Archcombined.jpg)

The pseudocode is described here:

![Pseud](/readme/pseudocode.png)

## Demo of LSI Workflow

Once the LSI architecture is trained, 


![ref_all](/readme/test_sigs_ref_recons_demo.png)

![zds](/readme/test_zds_demo.png)

![zms](/readme/test_zms_demo.png)



![m_pert](/readme/test_m_pert_demo.png)

![d_pert](/readme/test_d_pert_demo.png)


