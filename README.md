# latent-space-inversion-lsi

This repository supplements [**H052-08 - Latent-Space Inversion (LSI) for Subsurface Flow Model Calibration with Physics-Informed Autoencoding**](https://agu.confex.com/agu/fm20/meetingapp.cgi/Paper/753910) as presented at the American Geophysical Union Fall Meeting 2020 (AGUFM2020) and [Latent-Space Inversion (LSI): A Deep Learning Framework for Inverse Mapping of Subsurface Flow Data (COMG)](https://scholar.google.com/citations?user=mQUFzL8AAAAJ&hl=en#d=gs_md_cita-d&u=%2Fcitations%3Fview_op%3Dview_citation%26hl%3Den%26user%3DmQUFzL8AAAAJ%26citation_for_view%3DmQUFzL8AAAAJ%3AYsMSGLbcyi4C%26tzom%3D480). 

```
latent-space-inversion 
│
└─── mnist
│   
└─── 2d-fluvial
```

Demos based on the MNIST dataset and a 2D fluvial field dataset (see folder structure) are archived in this repository.

The AGUFM2020 [video](https://vimeo.com/495980342) presentation can be viewed below:

[![Video](/readme/thumb2.png)](https://vimeo.com/495980342)

## Workflow

LSI performs simultaneous dimensionality reduction (by extracting salient spatial features from **M** and temporal features from **D**) and inverse mapping (by mapping the salient features in **D** to **M**, i.e. latent spaces **z_d** and **z_m**). The architecture is composed of dual autoencoders connected with a regression model as shown below and is trained jointly. Since the latent spaces **z_d** and **z_m** correspond to meaningful spatial and temporal variations in **D** and **M** respectively, they can be explored to obtain an ensemble of relevant inversion solutions. The sampled points in the data latent space can be inversely mapped to the model latent space and the model decoder can be used to decode the sampled points around **z_m** to obtain the ensemble of relevant inversion solutions.

![Workflow](/readme/wflow.jpg)

In practical applications, observed **d** can be noisy and LSI helps us to quickly obtain the ensemble of relevant inversion solutions that can be accepted within the noise level, as well as understand the variations of spatial features within the ensemble to improve the predictive power of the inversion solutions.