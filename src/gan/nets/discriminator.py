import numpy as np
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator network for the GAN. A feedforward neural network architecture. Outputs a score
    for each image of the batch. The score represents how much the discriminator thinks the image
    comes from the real or generated distribution.

    :param img_shape: tuple. The shape of input images (channels, width, height). Needed to
    calculate the number of nodes for the input layer.

    :method forward: Defines the forward pass of the discriminator.
        :param img: torch.Tensor. The input tensor containing a batch of images.

        :return: torch.Tensor. The output tensor containing the scores for each image of the batch.
    """
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)  # Flatten the input image
        validity = self.model(img_flat)  # Pass the image through the network
        # Return the output of the last layer (1 output node, one float number is the score)
        return validity
