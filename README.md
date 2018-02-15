# Variational Autoencoders Project
In this project we compare various Autoencoder architectures.
We look at each model's reconstruction error with/without noisy test input.
This error is measured by the test log-likelihood.

Furthermore, we look at the latent space representation of each model in the case of 2-dimensional encodings.
This way we obtain a scatter plot of the latent representation.
We also reconstruct the latent space using uniform samples, which helps us to spot encodings that the decoder struggles to decode.

## Dataset
All our experiments are run on the MNIST dataset of handwritten digits.
We therefore encode 2 dimensional b/w images.
For each image we have 28x28 real-valued inputs between 0 and 1.
The dataset can directly be downloaded using the tensorflow python library.

## Project Structure
* **DUMPS:** Tensorflow dumps of trained models.
* **images:**
    * Reconstructions from (noisy) test inputs
    * Latent space representations in and uniformly sampled reconstructions in case of 2-dimensional encodings
* **logs:** TensorBoard logs for visual inspection
* **MNIST_data:** MNIST dataset downloaded using TensorFlow
* **models:** Python code for all the TensorFlow implementations (they all follow the same implementations design)
    * bayesian_autoencoder.py: Vanilla Bayesian Neural Network Autoencoder
    * bayesian_conv_autoencoder.py: Bayesian Neural Network Autoencoder using Convolutions
    * bayesian_vae_artificial.py: Bayesian Neural Network Autoencoder with intermediate gaussian sampling to mimic VAE behaviour.
    * bayesian_vae.py: An attempt of implementing a variational autoencoder with bayesian weights
    * GAN.py: Adversarial Autoencoder
    * variational_autoencoder.py: Variational Autoencoder (according to Kingma & Welling)
    * variational_conv_autoencoder.py: Variational Autoencoder using convolutions
* **Presentation:** Contains the final presentation of the project
* **Root directory:** Contains all the jupyter notebooks

## Jupyter Notebooks
Each notebook contains runs for one specific model from the models folder.
The runs have aligned architectures and plots of the latent space.

* Bayesian Neural Network.ipynb: TensorFlow experiments with a generic bayesian neural network (no autoencoder)
* Bayesian-VAE.ipynb: Notebook for bayesian_vae.py
* BNN-Autoencoder.ipynb: Notebook for bayesian_autoencoder.py
* BNN-Autoencoder-Convolutions.ipynb: Notebook for bayesian_conv_autoencoder.py
* BNN-VAE-Artificial.ipynb: Notebook for bayesian_vae_artificial.py
* GAN.ipynb: Notebook for GAN.py
* PlainAE.ipynb: Vanilla Autoencoder (nothing bayesian)
* PlainCAE.ipynb: Autoencoder with convolutions (non-bayesian)

## Autoencoder Model Structure
* __init__: Sets up the weights and the computational graph for the model
* sample_from_W: Samples from the variational distribution over the weights
* encode: The encoder function of the autoencoder
* decode: The decoder function of the autoencoder
* feedforward: Describes the full model architecture using _encode_ and _decode_
* get_ll: The log-likelihood per sample
* get_ell: The expected log-likelihood approximated by monte carlo sampling
* get_kl: The KL-divergence between the variational distribution and the prior
* get_nelbo: Get the negative Eidence Lower Bound (ELBO) - the optimization objective
* learn: The training procedure
* benchmark: Compute validation/test error
* serialize: Store the trained model on disk
* restore: Restore a trained model from disk
* get_weights: Return the models weights for inspection
* plot_enc_dec: Plot inputs vs. their reconstructions
* plot_noisy_recon: Plot inputs with added gaussian noise and their reconstructions
* plot_latent_repr: Plot the encodings of the test data if 2-dimensional
* plot_latent_recon: Plot a uniformly sampled reconstruction of the encodings of the test data
