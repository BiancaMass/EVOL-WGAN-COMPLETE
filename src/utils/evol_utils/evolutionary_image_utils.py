# import torch
import random
import torchvision.transforms.functional as TF


def crop_images_for_evol(dataloader, patch_position, patch_height, n_batches, device):
    """
    Takes a dataloader (training images from mnist) and crops them to the size of one
    patch, only keeping a single patch.
    Returns a list containing one patch from each original image (or as many as indicated by the
    number of batches).

    :param dataloader: Dataloader. Dataset to extract the patches.
    :param patch_position: int or str. If int, the index of the first row of the desired patch.
        If string, must be equal to 'random' in which case the patch is chosen at random. e.g.,
        if set to 4, it will take the patch starting from the 4th row for all the images. Random will
        take a different one for every image.
    :param patch_height: int. The height of one patch.
    :param n_batches: int. The max number of image batches to preload.
    :param device: the device that is currently being used for the computations (cpu or gpu)
    :return: list. A list of image patches og length (n_batches x batch_size)
    """
    # Images are in shape [batch, channels, rows, cols]

    cropped_real_image_batches = []

    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        image_height = real_images.size(2)
        image_width = real_images.size(3)

        for j in range(batch_size):
            if isinstance(patch_position, int):
                top = patch_position
                if top + patch_height > image_height:
                    top = 0  # just take top patch in that case
            elif patch_position == 'random':
                max_start_point = image_height - patch_height
                top = random.randint(0, max_start_point)
            else:
                raise ValueError("Invalid patch_position: must be 'random' or an integer.")

            cropped_image = TF.crop(real_images[j], top, 0, patch_height, image_width)
            cropped_real_image_batches.append(cropped_image.to(device))

        if i + 1 == n_batches:
            break

    # Tried a more efficient way (above) using crop function
    # cropped_real_image_batches = []
    #
    # if isinstance(patch_position, int):
    #     for i, (real_images, _) in enumerate(dataloader):
    #         start_row = patch_position
    #         end_row = start_row + patch_height
    #         if end_row < real_images.size(2):
    #             cropped_images = real_images[:, :, start_row:end_row, :]
    #             cropped_real_image_batches.append(cropped_images.to(device))
    #         else:  # just take the top patch in this case
    #             start_row = 0
    #             end_row = start_row + patch_height
    #             cropped_images = real_images[:, :, start_row:end_row, :]
    #             cropped_real_image_batches.append(cropped_images.to(device))
    #         if i + 1 == n_batches:
    #             break
    #
    # if patch_position == 'random':
    #     for i, (real_images, _) in enumerate(dataloader):
    #         batch_size = real_images.size(0)
    #         image_height = real_images.size(2)
    #         max_start_point = image_height - patch_height
    #         start_rows = torch.randint(0, max_start_point, (batch_size,), device=device)
    #         # Create a tensor of start indices for cropping
    #         ranges = [start_rows[j] + torch.arange(patch_height, device=device) for j in range(batch_size)]
    #         # Create ranges for each image using the starts_tensor
    #         cropped_images = torch.stack([real_images[j, :, ranges[j], :] for j in range(batch_size)], dim=0)
    #         cropped_real_image_batches.append(cropped_images.to(device))
    #         if i + 1 == n_batches:
    #             break

    return cropped_real_image_batches
