# Variational Autoencoder (VAE) for Generating Faces

This project presents a Variational Autoencoder (VAE) trained on the CelebA dataset for 50 epochs. The model can generate novel images of faces, demonstrating impressive results. Two additional features are also provided: a function to display the distribution of the latent space variables and another to manipulate novel images by performing arithmetic in the latent space.

## Table of Contents

- [Background](#background)
- [Project Description](#project-description)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Sources](#sources)

## Background

A Variational Autoencoder (VAE) is a type of generative model that's excellent for learning a compressed representation of input data. As seen below, It achieves this by encoding the data into a lower-dimensional latent space and then decoding it back. This project uses a VAE to generate novel face images. 

![image](https://github.com/DimensionDweller/VAE_Implimentation/assets/75709283/13374597-771f-4fe9-8dd3-f946d6b70189)


## Project Description

The model is trained on the CelebA dataset, a large-scale face attributes dataset with more than 200,000 celebrity images, each with 40 attribute annotations. The training was performed for 50 epochs.

This project also includes two additional features:
- A function to display the distribution of the latent space variables. This helps to understand the distribution of the latent space learned by the VAE.
- A function to manipulate novel images by performing arithmetic in the latent space. This feature allows users to generate novel faces with specified attributes by manipulating the latent variables.


## Model Architecture

The architecture of the Variational Autoencoder (VAE) is crucial to its performance. It consists of an encoder, a decoder, and a fully connected middle layer that connects the two. The encoder and decoder are both composed of several layers of Convolutional Neural Networks (CNNs). The architecture of the model can be described as follows:

### Encoder
The encoder part of the model takes the input image and encodes it into a lower-dimensional latent space. It consists of a series of convolutional layers, each followed by a batch normalization layer and a LeakyReLU activation function. The output of these layers is flattened and passed through two separate fully connected layers to get the mean ($\mu$) and the logarithm of the variance $\log(\sigma^2)$ of the latent distribution.

### Reparametrization Trick
The reparametrization trick is employed to sample from the latent distribution without having to backpropagate through the random node. This is achieved by generating a random tensor ($\epsilon$) with the same size as $\sigma$ and calculating the sample $z = \mu + \sigma \cdot \epsilon$.

### Decoder

The decoder takes the sampled latent vector and decodes it back into an image. The latent vector is first passed through a fully connected layer and reshaped to match the output shape of the encoder. It is then passed through several transposed convolutional layers (also known as deconvolutional layers in some contexts), each followed by a LeakyReLU activation function. The final layer uses a sigmoid activation function to ensure the output values are between 0 and 1.

Below is the detailed architecture:

```
VAE(
  (encoder): Sequential(
    (0): Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.01)
    ...
    (10): LeakyReLU(negative_slope=0.01)
  )
  (fc_mu): Linear(in_features=16384, out_features=200, bias=True)
  (fc_logvar): Linear(in_features=16384, out_features=200, bias=True)
  (fc_decode): Linear(in_features=200, out_features=16384, bias=True)
  (decoder): Sequential(
    (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.01)
    ...
    (6): ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (7): Sigmoid()
  )
)
```

This architecture works well because CNNs are particularly good at handling image data, and the use of transposed convolutions allows the model to generate images that preserve spatial information from the latent space. The use of the LeakyReLU activation function helps to mitigate the vanishing gradients problem during training.

### Loss Function

The loss function used for training the VAE is composed of two terms:

1. **Reconstruction loss (MSE Loss)**: This is the mean squared error (MSE) between the reconstructed image and the original image. It encourages the VAE to accurately reconstruct the input images.

   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (X_{\text{original}}^i - X_{\text{reconstructed}}^i)^2$$

2. **Kullback-Leibler divergence (KL divergence)**: This is a measure of how one probability distribution is different from a second, reference probability distribution. In this case, it measures the divergence between the latent distribution and a standard normal distribution. It encourages the latent variables to be distributed as a standard normal distribution, which is a critical assumption of the VAE.

   Apologies for the oversight. Here's the KL divergence term in LaTeX format, which should be compatible with GitHub Markdown:


   $$\text{KL}(q(z|x) \| p(z)) = -\frac{1}{2} \sum_{i=1}^{d} \left(1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2\right)$$

The final loss function is a weighted sum of these two terms. The weighting factor `BETA` is a hyperparameter that determines the balance between the reconstruction loss and the KL divergence.

## Results

![image](https://github.com/DimensionDweller/VAE_Implimentation/assets/75709283/37f83e55-d0cc-4ab8-a044-2afff01beb69)


---

## Latent Space Distribution

![image](https://github.com/DimensionDweller/VAE_Implimentation/assets/75709283/984ee222-1070-4da2-9b78-20b1cc282bad)


One of the key aspects of Variational Autoencoders, which distinguishes them from regular autoencoders, is the "reparametrization trick", an elegant method to allow backpropagation through a random node.

In a VAE, the encoder doesn't output an explicit code or representation. Instead, it produces parameters (means and variances) of a set of Gaussian distributions. The actual latent code is then sampled from these distributions. This is where the "reparametrization trick" comes in: instead of sampling from the Gaussian distribution directly, which is a stochastic process and thus not differentiable, we sample from a unit Gaussian and then reparametrize to obtain the sample from the desired distribution. This process allows us to retain differentiability in the network, which is essential for backpropagation.

The latent space distributions shown above demonstrate this property. Ideally, each of these distributions should resemble a standard Gaussian distribution, which is a result of the VAE's training objective. Effectively, this is the Kullback-Leibler (KL) divergence term in action.

## Latent Space Manipulation

Variational Autoencoders (VAEs) not only have the ability to generate new images, but they can also learn a meaningful structure in the latent space that we can explore and manipulate. This is made possible by the reparametrization trick, which allows the VAE to learn a continuous distribution in the latent space. As a result, similar images are encoded to nearby points, facilitating meaningful transformations.

Let's explore this fascinating feature of VAEs through an experiment where we manipulate attributes like "smiling" and "wearing glasses" in the generated images. We've trained our model on the CelebA dataset, which includes attribute labels for each image, making it possible for us to perform these manipulations.

### Making the model smile

In the image grid below, you can see a series of faces that have been manipulated to varying degrees to add or remove a smile. This was achieved by moving along the direction in the latent space that the model has associated with the "smiling" attribute. Negative values make the face less happy, while positive values make the face more happy.

![image](https://github.com/DimensionDweller/VAE_Implimentation/assets/75709283/de6bc9db-363d-41b1-a6a0-f883872c82a8)

### Adding glasses

Similarly, we can manipulate the "eyeglasses" attribute to add or remove glasses from a face. Again, the model has learned to associate this attribute with a certain direction in the latent space, and by moving along this direction, we can control the presence of glasses.

![image](https://github.com/DimensionDweller/VAE_Implimentation/assets/75709283/3b9472c7-76fd-42c9-86e0-625f3923309e)


This is the true beauty of Variational Autoencoders. They not only have the capacity to generate new images, but they also learn a meaningful structure in the latent space that corresponds to semantically meaningful transformations in the data space. When the model is trained on a dataset with labeled attributes like CelebA, these transformations can correspond to identifiable features such as a smile or a pair of glasses. This makes it possible to manipulate these features in novel images, demonstrating the creative potential of VAEs.

## Usage

Firstly, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/DimensionDweller/VAE_Implimentation.git
```

Navigate to the directory of the project:

```bash
cd https://github.com/DimensionDweller/VAE_Implimentation.git
```

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

Run the following command to start training the model:

```bash
python train.py
```

You can adjust the hyperparameters of the model by modifying the `config.py` file.


Please note that you may need a machine with a GPU to train the model in a reasonable amount of time. The code is set up to use a GPU if one is available, and will otherwise fall back to using a CPU.

## Future Work and Conclusion

Variational Autoencoders (VAEs) have played an instrumental role in the development of generative models. They were one of the first models to successfully combine deep learning with variational inference, thereby enabling the generation of complex and high-quality images. VAEs introduced the idea of a structured latent space where similar images are encoded to nearby points, paving the way for more advanced models.

In recent years, newer models such as Generative Adversarial Networks (GANs) and more recently, Stable Diffusion models, have pushed the boundaries of what's possible in image generation. These models have been able to generate incredibly realistic images, often indistinguishable from real photos. However, these advancements would not have been possible without the foundational work done by VAEs.

One of the key concepts introduced by VAEs, the idea of a structured latent space, is still actively used in modern models. It allows us not only to generate new images, but also to explore and manipulate the latent space. As demonstrated in this project, this can lead to fascinating results, such as the ability to add or remove a smile from a face.

In future work, it would be interesting to explore how these concepts can be extended and applied in other domains. For instance, similar techniques could be used to manipulate other types of data, such as text or audio. There is also a lot of potential in combining the strengths of different types of generative models, such as the structured latent space of VAEs and the high-quality generation capabilities of GANs or Stable Diffusion models.

To conclude, while VAEs may no longer be state-of-the-art in terms of image quality, they remain a powerful tool for understanding and manipulating high-dimensional data. The concepts introduced by VAEs have had a profound impact on the field of generative models, and will undoubtedly continue to influence future developments.


## Sources

Fan, H., Su, H., & Guibas, L. J. (2016). Deep Feature Consistent Variational Autoencoder. Retrieved from https://arxiv.org/abs/1610.00291

Pu, Y., Gan, Z., Henao, R., Yuan, X., Li, C., Stevens, A., & Carin, L. (2016). Variational Autoencoder for Deep Learning of Images, Labels and Captions. Retrieved from https://arxiv.org/abs/1610.03483

Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2018). Understanding disentangling in β-VAE. Retrieved from https://arxiv.org/abs/1804.03599

Kostrikov, I., & Mnih, A. (2018). Variational Autoencoder with Arbitrary Conditioning. Retrieved from https://arxiv.org/abs/1806.02382

Kumar, A., Sattigeri, P., Balakrishnan, A., Goyal, A., Zhou, Y., Bhardwaj, O., Steinbach, M., & Kumar, V. (2018). Learning Latent Subspaces in Variational Autoencoders. Retrieved from https://arxiv.org/abs/1812.06190


