import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader


def dataloader_mnist(num_workers, data_dir='./datasets', image_side=None, classes=None,
                     train=True, evol_batch_size=100, gan_batch_size=25,
                     gan_train_size=800, gan_val_size=200, evol_train_size=300, evol_val_size=100):
    """
    Load the MNIST dataset. Download it if not present locally, normalize pixel values to [-1, 1].
    Optionally resize images. Returns training and validation sets for the evolutionary algorithm
    and GAN.

    :param num_workers: int. Number of parallel processes for loading and preprocessing images.
    :param data_dir: str. Path to dataset directory. Defaults to './datasets'.
    :param image_side: tuple or None. Desired image size. Default MNIST size is 28x28.
    :param classes: list. Classes to load, e.g., [0, 1].
    :param train: bool. Load from training set (True) or test set (False).
    :param evol_batch_size: int. Batch size for evolutionary training. Defaults to 100.
    :param gan_batch_size: int. Batch size for GAN training. Defaults to 25.
    :param gan_train_size: Number of samples per class for GAN training. Defaults to 800.
    :param gan_val_size: Number of samples per class for GAN validation. Defaults to 200.
    :param evol_train_size: Number of samples per class for evolutionary training. Defaults to 300.
    :param evol_val_size: Number of samples per class for evolutionary validation. Defaults to 100.

    :return: Four DataLoader objects for the specified subsets of the MNIST dataset:
             train_gan_loader, val_gan_loader, train_evo_loader, val_evo_loader.
    """
    # Define the transformations to apply to the data before loading.
    if image_side is not None:
        transform = transforms.Compose([transforms.Resize(image_side),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST dataset.
    full_dataset = torchvision.datasets.MNIST(root=data_dir,
                                              train=train,
                                              download=True,
                                              transform=transform)

    classes = classes if classes is not None else [0, 1]
    class_indices = [i for i, (_, label) in enumerate(full_dataset) if label in classes]
    class_dataset = Subset(full_dataset, class_indices)

    # subset points:
    a = evol_train_size
    b = a + evol_val_size
    c = b + gan_train_size
    d = c + gan_val_size

    # Split indices for GAN and evolutionary training/validation
    evol_train_indices = []
    evol_val_indices = []

    gan_train_indices = []
    gan_val_indices = []

    for cl in classes:
        current_class_indices = [i for i, (_, label) in enumerate(class_dataset) if label == cl]
        evol_train_indices.extend(current_class_indices[:a])
        evol_val_indices.extend(current_class_indices[a:b])
        gan_train_indices.extend(current_class_indices[b:c])
        gan_val_indices.extend(current_class_indices[c:d])

    # Create subsets
    train_evol_dataset = Subset(class_dataset, evol_train_indices)
    val_evol_dataset = Subset(class_dataset, evol_val_indices)
    train_gan_dataset = Subset(class_dataset, gan_train_indices)
    val_gan_dataset = Subset(class_dataset, gan_val_indices)

    # Create data loaders
    train_gan_loader = DataLoader(train_gan_dataset,
                                  batch_size=gan_batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    val_gan_loader = DataLoader(val_gan_dataset,
                                batch_size=gan_batch_size,
                                shuffle=False,
                                num_workers=num_workers)

    train_evo_loader = DataLoader(train_evol_dataset,
                                  batch_size=evol_batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    val_evo_loader = DataLoader(val_evol_dataset,
                                batch_size=evol_batch_size,
                                shuffle=False,
                                num_workers=num_workers)

    return train_gan_loader, val_gan_loader, train_evo_loader, val_evo_loader
