from data_loader_MNIST import dataloader_mnist


def crop_images_for_evol(dataloader, patch_height, n_batches, device):
    """
    # TODO: update documentation
    Pre-loads a specified number of image batches from the mnist dataset, with the given
    classes. Crops the images to only keep as many pixels as there are in one patch (top
    patch) and discards the rest. Adds the cropped image to a list. This list of cropped
    image batches is then used by the cost function to guide the evolutionary ansatz search
    (distance between these pre-loaded images and generated images is minimized).

    :param dataloader:
    :param n_batches: int. The number of image batches to preload.
    :return: list. A list of preloaded and cropped image batches.
    """
    # TODO: currently only loads the top patch if patches. Might be okay but think about it.
    cropped_real_image_batches = []
    for i, (real_images, _) in enumerate(dataloader):
        target_height = patch_height
        cropped_images = real_images[:, :, :target_height, :]
        cropped_real_image_batches.append(cropped_images.to(device))
        if i + 1 == n_batches:  # Stop after preloading n_batches
            break
    return cropped_real_image_batches