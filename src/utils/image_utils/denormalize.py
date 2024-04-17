
def denorm(x):
    """
    Changes x's values from the range [-1, 1] to the range [0, 1] before saving the image.
    When importing MNIST I normalize to range [-1, 1] for training the GAN. This brings it back to
    normal pixel value interval for grayscale images [0, 1] before saving to local storage.

    First checks if the input tensor or value `x` is between [-1, 1]. If not, Assertion Error.
    Args:
        x (torch.Tensor): The tensor or value to be denormalized.
    Returns:
        torch.Tensor: The denormalized tensor or value, clamped between 0 and 1.
    """
    # Make sure x is within [-1, 1]
    assert x.min() >= -1 and x.max() <= 1, "Input value(s) must be within the range [-1, 1]."
    # Map values from [-1, 1] to [0, 1]
    out = (x + 1) / 2
    # Clamping as a safety extra step for e.g., floating point approximations/overflow
    # You cannot just clamp because e.g., negative values would all be set to 1 (information loss)
    return out.clamp(0, 1)
