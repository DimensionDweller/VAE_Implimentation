from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(BATCH_SIZE):
    """
    Loads the CelebA dataset and returns DataLoader objects for the train and test sets.

    Args:
    - BATCH_SIZE (int): The batch size to use when creating the DataLoaders.

    Returns:
    - trainloader (DataLoader): A DataLoader for the training data.
    - testloader (DataLoader): A DataLoader for the test data.
    """
    # Compose transformations to apply to the images
    transform = transforms.Compose([
        transforms.CenterCrop(128),  # Crop the center of the images
        transforms.ToTensor()  # Convert the images to PyTorch tensors
    ])

    # Load the CelebA dataset
    # The 'download=True' argument will download the dataset if it's not found in the specified directory
    trainset = datasets.CelebA(root='./data', split='train', target_type='attr', download=True, transform=transform)
    testset = datasets.CelebA(root='./data', split='test', target_type='attr', download=True, transform=transform)

    # Create DataLoaders for the train and test sets
    # The 'shuffle=True' argument shuffles the data at each epoch
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    
    return trainloader, testloader
