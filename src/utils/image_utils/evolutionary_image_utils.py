

def preload_real_images_batches(self, n_batches=10):
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
    dataset = select_from_dataset(load_mnist(image_size=self.patch_width), 1000, self.classes)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
    real_images_batches = []
    for i, (real_images, _) in enumerate(dataloader):
        target_height = self.patch_height
        cropped_images = real_images[:, :, :target_height, :]
        real_images_batches.append(cropped_images.to(self.device))
        if i + 1 == n_batches:  # Stop after preloading n_batches
            break
    return real_images_batches