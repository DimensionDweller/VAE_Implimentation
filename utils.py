import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import random


def vae_loss(recon_x, x, mu, logvar, BETA):
  """
    Computes the loss for a VAE. The loss is the sum of the reconstruction loss (MSE) and 
    the KL divergence which encourages the latent variables to be distributed according to a unit Gaussian.
    
    Args:
    - recon_x (tensor): The reconstructed images.
    - x (tensor): The original images.
    - mu (tensor): The vector of means of the latent distribution.
    - logvar (tensor): The vector of log variances of the latent distribution.
    - BETA (float): The coefficient for the reconstruction loss.

    Returns:
    - A dictionary containing the total loss, reconstruction loss, and KL divergence.
    """

    MSE = F.mse_loss(recon_x.view(-1, 3*128*128), x.view(-1, 3*128*128), reduction='sum') / x.size(0)
    
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    loss = BETA * MSE + KLD
    
    return {'loss': loss, 'Reconstruction_Loss': MSE, 'KLD': KLD}



def plot_latent_distribution(model, dataloader, device, num_variables=50):
  """
    Plots the distribution of the latent variables. Can be used to visualize the distribution of the latent variables.
    
    Args:
    - model (nn.Module): The VAE model.
    - dataloader (DataLoader): The DataLoader for the dataset.
    - device (device): The device type (CPU or GPU).
    - num_variables (int, optional): The number of latent variables to plot. Default is 50.
    """

    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            mu, logvar = model.encode(data)
            z = model.reparametrize(mu, logvar)
            break  # only use one batch
    
    z = z.detach().cpu().numpy()  # convert to numpy array
    fig, axs = plt.subplots(5, 10, figsize=(20, 10))
    
    for i, ax in enumerate(axs.flatten()):
        if i >= num_variables:
            break
        ax.hist(z[:, i], bins=20, density=True)
        ax.set_title(f'Latent var {i+1}')
    
    plt.tight_layout()
    plt.show()


def make_smile(model, dataloader, device, index, weights, smiling=True, example_image=None, random_example=False):
      """
    Modifies the expression of the faces in the dataset to make them smile or not smile.
    
    Args:
    - model (nn.Module): The VAE model.
    - dataloader (DataLoader): The DataLoader for the dataset.
    - device (device): The device type (CPU or GPU).
    - index (int): The index of the attribute for smiling in the dataset (or any other attribute you would like to manipulate).
    - weights (list of float): The weights for the difference in the average latent vector of smiling and not smiling faces.
    - smiling (bool, optional): If True, the faces are made to smile. If False, the faces are made to not smile. Default is True.
    - example_image (tensor, optional): An example image to modify. If None, the average face is used. Default is None.
    - random_example (bool, optional): If True, a random image from the dataset is used as the example image. If False, the example_image is used if it is not None. Default is False.
    """
  
    model.eval()
    z_smiling = []
    z_not_smiling = []
    print("Collecting smiling and non-smiling faces...")

    with torch.no_grad():
        for i, (data, attrs) in enumerate(dataloader):
            data = data.to(device)
            mu, logvar = model.encode(data)
            z = model.reparametrize(mu, logvar)
            z = z.detach().cpu().numpy()  # convert to numpy array
            attrs = attrs[:, index]  # get the attribute for smiling

            z_smiling.extend(z[attrs == 1])
            z_not_smiling.extend(z[attrs == 0])

    z_smiling = np.array(z_smiling)
    z_not_smiling = np.array(z_not_smiling)

    z_smiling_avg = np.mean(z_smiling, axis=0)
    z_not_smiling_avg = np.mean(z_not_smiling, axis=0)

    difference = z_smiling_avg - z_not_smiling_avg

    fig, axs = plt.subplots(4, 2, figsize=(10, 20))

    if random_example:
        print("Grabbing random face...")
        example_image, _ = next(iter(dataloader))  # get a batch of images
        example_image = example_image[random.randint(0, example_image.size(0) - 1)]  # take a random image in the batch

    print("Morphing faces in latent space...")
    for i, weight in enumerate(weights):
        if example_image is not None:
            mu, logvar = model.encode(example_image.unsqueeze(0).to(device))
            z_example = model.reparametrize(mu, logvar)
            z_example = z_example.detach().cpu().numpy()
            z_new = z_example + weight * difference if smiling else z_example - weight * difference
        else:
            z_new = z_smiling_avg + weight * difference if smiling else z_not_smiling_avg - weight * difference
        z_new = torch.from_numpy(z_new).float().unsqueeze(0).to(device)
        new_img = model.decode(z_new).view(3, 128, 128).detach().cpu()

        axs[i // 2, i % 2].imshow(np.transpose(new_img.numpy(), (1, 2, 0)))
        axs[i // 2, i % 2].set_title(f'Weight: {weight}')
        axs[i // 2, i % 2].axis('off')

    plt.tight_layout()
    plt.show()
