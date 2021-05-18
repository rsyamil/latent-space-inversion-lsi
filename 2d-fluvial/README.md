# latent-space-inversion

## Prerequisites

The dataset used in this demo repository is a small subset of 2D non-Gaussian images consisting of channelized (fluvial) geologic features. There are five possible geologic (i.e. based on meandering and anastomosing fluvial depositional environment) prior scenarios in the dataset. Refer to the paper for more information on this dataset. The images/prior models are denoted as **X**, with the forward model being a linear operator **G** and the resulting simulated responses denoted as **Y**. The physical system is represented as **Y=G(X)** or **D=G(M)**. We are interested in learning the inverse mapping **M=G'(D)** which is not trivial if the **M** is non-Gaussian (which is the case with the fluvial models) and G is nonlinear (in this demo we assume a linear operator). Such complex mapping (also known as history-matching) may result in solutions that are non-unique with features that may not be consistent. LSI seeks to address those challenges and is an extension of [my early work](https://link.springer.com/article/10.1007/s10596-020-09971-4) that has more descriptions of the nature of this engineering problem.

## Demo of LSI Workflow

The trained LSI architecture is used to provide predictions of **m** for any given (and unseen) **d**. In this demonstration, assume the following reference case of anastomosing fluvial model with azimuth 135. The first left plot in the figure below represents the observed **d** and its reconstruction. The second plot shows the same, in scatter form. The third plot is the reference **m** and the fourth plot shows the discretized prediction of LSI. 

![ref_all](/2d-fluvial/readme/test_sigs_ref_recons_demo.png)

Since the latent spaces **z_d** and **z_m** correspond to meaningful spatial and temporal variations in **D** and **M** respectively, they can be explored to obtain an ensemble of relevant inversion solutions. For example, the histograms below show **z_d** and the red line represents the latent variables **z_d** of the observed **d**. We can sample points around **z_d**, shown as the black unfilled bars denoted as "Pert".

![zds](/2d-fluvial/readme/test_zds_demo.png)

Similarly, the histograms below show **z_m** and the red line represents the  latent variables **z_m** of the reference **m**. The sampled points in the data latent space can be inversely mapped to the model latent space, shown as the black unfilled bars denoted as "Pert".

![zms](/2d-fluvial/readme/test_zms_demo.png)

The model decoder can be used to decode the sampled points around **z_m** to obtain the ensemble of relevant inversion solutions, as shown below. The inversion solutions show variations in which the anastomosing fluvial model can still reproduce the observed **d**. 

![m_pert](/2d-fluvial/readme/test_m_pert_demo.png)

The ensemble of relevant inversion solutions is fed to the operator **G** (i.e. **D=G(M)**) to validate if the solutions can sufficiently reproduce the observed **d**. In the figure below, gray points/lines represent the entire testing dataset and the blue points/lines represent the simulated **D** from the ensemble, where a significant reduction in uncertainty is obtained.

![d_pert](/2d-fluvial/readme/test_d_pert_demo.png)

In practical applications, observed **d** can be noisy and LSI helps us to quickly obtain the ensemble of relevant inversion solutions that can be accepted within the noise level, as well as understand the variations of spatial features within the ensemble to improve the predictive power of the inversion solutions.


