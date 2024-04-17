from data_loader_MNIST import dataloader_mnist


def load_and_crop_images_for_evo(self, n_batches=10):
    """
    Pre-loads a specified number of image batches from the mnist dataset, with the given
    classes. Crops the images to only keep as many pixels as there are in one patch (top
    patch) and discards the rest. Adds the cropped image to a list. This list of cropped
    image batches is then used by the cost function to guide the evolutionary ansatz search
    (distance between these pre-loaded images and generated images is minimized).

    :param n_batches: int. The number of image batches to preload. Defaults to 10.
    :return: list. A list of preloaded and cropped image batches.
    """
    # TODO: currently only loads the top patch if patches. Might be okay but think about it.
    print("Pre-loading real image batches")
    dataloader = dataloader_mnist(num_workers, file_location='./datasets', image_size=None,
                                  classes=None, samples_per_class=1000, train=True, batch_size=25)
    real_images_batches = []
    for i, (real_images, _) in enumerate(dataloader):
        target_height = self.patch_height
        cropped_images = real_images[:, :, :target_height, :]
        real_images_batches.append(cropped_images.to(self.device))
        if i + 1 == n_batches:  # Stop after preloading n_batches
            break
    return real_images_batches