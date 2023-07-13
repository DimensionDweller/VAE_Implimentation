from tqdm import tqdm
import torch
import wandb

def train(model, optimizer, train_loader, device, start_epoch, num_epochs, BETA, sample_size=8, log_wandb=True, checkpoint_path=None):
  """
    Trains a VAE model.

    Args:
    - model (nn.Module): The VAE model.
    - optimizer (Optimizer): The optimizer for training.
    - train_loader (DataLoader): The DataLoader for the training set.
    - device (device): The device type (CPU or GPU).
    - start_epoch (int): The starting epoch number.
    - num_epochs (int): The total number of epochs to train.
    - BETA (float): The coefficient for the reconstruction loss.
    - sample_size (int, optional): The number of images to sample for visualization during training. Default is 8.
    - log_wandb (bool, optional): If True, logs to Weights & Biases are enabled. If False, no logging occurs. Default is True.
    - checkpoint_path (str, optional): The path to a checkpoint file to load. If None, no checkpoint is loaded. Default is None.

    Returns:
    - The average loss per data point over the training set.
    """
    
    if checkpoint_path is not None and log_wandb:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loaded model from checkpoint')

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        total_loss = 0
        recon_loss = 0
        kld_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss_results = vae_loss(recon_batch, data, mu, logvar, BETA)
            loss = loss_results['loss']
            loss.backward()
            total_loss += loss.item()
            recon_loss += loss_results['Reconstruction_Loss'].item()
            kld_loss += loss_results['KLD'].item()
            optimizer.step()

            if batch_idx == 0:
                n = min(data.size(0), sample_size)
                comparison = torch.cat([data[:n], recon_batch.view(data.size(0), 3, 128, 128)[:n]])
                images = comparison.cpu()
                recon_images = recon_batch.view(data.size(0), 3, 128, 128).cpu()
                
        if log_wandb:
            wandb.log({
                'Total Loss': total_loss/len(train_loader),
                'Reconstruction Loss': recon_loss/len(train_loader),
                'KLD': kld_loss/len(train_loader),
                "original images": [wandb.Image(image) for image in images],
                "reconstructed images": [wandb.Image(image) for image in recon_images]
            })

        with torch.no_grad():
            latent_samples = torch.randn(sample_size, model.latent_dim).to(device)
            generated_images = model.decode(latent_samples)
            generated_images = generated_images.cpu()
        
        if log_wandb:
            wandb.log({
                "generated images": [wandb.Image(image) for image in generated_images]
            })

        if log_wandb:
            checkpoint_path = f'checkpoint_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)

            wandb.save(checkpoint_path)

        print(f'Epoch {epoch+1}, Total Loss: {total_loss/len(train_loader):.4f}, '
            f'Reconstruction Loss: {recon_loss/len(train_loader):.4f}, '
            f'KLD: {kld_loss/len(train_loader):.4f}')

    if log_wandb:
        wandb.finish()

    return total_loss / len(train_loader.dataset)
