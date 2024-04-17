import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


def dataloader_mnist(num_workers, file_location='./datasets', image_size=None, classes=None,
                     samples_per_class=1000, train=True, batch_size=25):
    """
    Load the MNIST dataset. Downloads it if not already present on local storage.
    Normalizes the pixel values to the range [-1, 1]
    Resizes the images if image size is specified.
    Returns a subset of the data based on the specified classes and

    :param num_workers: int. Number of parallel processes for loading and preprocessing the images.
    :param file_location: str. Path to the directory where the dataset will be stored.
    Defaults to './datasets'.
    :param image_size: tuple or None: The desired size of the images. If None, images are not
    resized. Default MNIST size is 28x28.
    :param classes: list. Classes to load e.g., only [0,1]
    :param samples_per_class: int. how many samples to load per class
    :param train: whether to load from the training set (True) or test set (False).
    :param batch_size: int. Defaults to 25.

    :return: DataLoader. DataLoader for the specified subset of the MNIST dataset.
    """
    if classes is None:
        classes = [0, 1]

    # Define the transformations to apply to the data before loading.
    # Normalizing makes training easier and faster. 0.5, 0.5 is quite standard
    # If image size is specified, resize accordingly.
    if image_size is not None:
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST dataset, either train or test according to train argument
    full_dataset = torchvision.datasets.MNIST(root=file_location,
                                              train=train,
                                              download=True,
                                              transform=transform)

    # Create a subset with only the selected classes, limiting the number of samples per class
    # for efficiency
    counts = {class_id: 0 for class_id in classes}
    indices = []

    for idx, (image, label) in enumerate(full_dataset):
        # Select the indices we want to keep, to only load the desired subset of the data
        if counts.get(label, 0) < samples_per_class and label in classes:
            indices.append(idx)
            counts[label] += 1
            # Once all classes have enough samples, break
            if all(count >= samples_per_class for count in counts.values()):
                break

    dataset_subset = Subset(full_dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset_subset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)

    return dataloader


