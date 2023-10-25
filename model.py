import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_channel, latent_dim):
        """
        Initialize the VAE model.
        
        Args:
        - input_channel (int): The number of input channels.
        - latent_dim (int): The dimension of the latent space.
        """
        super(VAE, self).__init__()
        
        # Initialize input and latent dimensions
        self.input_channel = input_channel
        self.latent_dim = latent_dim
        

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channel, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        
        # These layers represent the mean and variance of the latent space distribution
        self.fc_mu = nn.Linear(256*8*8, self.latent_dim)
        self.fc_logvar = nn.Linear(256*8*8, self.latent_dim)
        
        # This layer transforms the latent vector back to the dimension of the encoder output
        self.fc_decode = nn.Linear(self.latent_dim, 256*8*8)
        

        # Convolutional transpose layers are used to upsample the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, self.input_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid function ensures the output is between 0 and 1

    def encode(self, x):
        # Encode the input image
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # flatten the output
        
        # Return the encoded mu and logvar
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu, logvar):
        # Reparametrize the encoded mu and logvar
        std = torch.exp(0.5*logvar)  # get the standard deviation
        eps = torch.randn_like(std)  # generate a random tensor with same size as std
        return mu + eps*std  # reparametrize
    
    def arithmetic(self, x1, x2):
        # Encode the input images
        mu1, logvar1 = self.encode(x1)
        z1 = self.reparametrize(mu1, logvar1)
        
        mu2, logvar2 = self.encode(x2)
        z2 = self.reparametrize(mu2, logvar2)
        
        # Perform arithmetic on z1 and z2
        z = z1 - z2  
        
        # Decode the result
        return self.decode(z)

    def reconstruct(self, x):
        # Reconstruct the input image
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z)
    
    def generate(self, batch_size):
        # Generate random samples
        z = torch.randn((batch_size, self.latent_dim)).to(device)
        return self.decode(z)
    
    def decode(self, z):
        # Decode the input tensor
        z = self.fc_decode(z)
        z = z.view(z.size(0), 256, 8, 8)  # reshape the tensor
        return self.decoder(z)
    
    def forward(self, x):
        # Forward pass through the model
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
        
    def init_weights(self):
        # Initialize the weights of the model
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    
