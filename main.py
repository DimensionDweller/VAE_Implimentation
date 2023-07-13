from models import VAE
from utils import vae_loss, plot_latent_distribution, make_smile
from train import train
from dataset import load_data
import torch.optim as optim
import torch

# constants
BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
trainloader, testloader = load_data(BATCH_SIZE)

# create model
model = VAE(input_channel=3, latent_dim=100).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.init_weights()

# train model
train(
    model=model,
    optimizer=optimizer,
    train_loader=trainloader,
    device=device,
    start_epoch=0,
    num_epochs=50,
    BETA=1,
    sample_size=8,
    log_wandb=True,
    checkpoint_path=None
)

# visualize results
plot_latent_distribution(model, trainloader, device, num_variables=50)
make_smile(model, trainloader, device, index=39, weights=[-3, -2, -1, 0, 1, 2, 3,4], smiling=True, random_example=True)
