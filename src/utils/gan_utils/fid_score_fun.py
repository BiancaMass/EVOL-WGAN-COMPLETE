from collections import OrderedDict

"This works but it is way too expensive, okay for the last computation but not to insert in the pipeline."
# TODO: check if there is a minimum required number of images for accurate computation.

from collections import OrderedDict
import torch
from torch import nn
from ignite.engine import Engine
from ignite.metrics import FID
from ignite.utils import manual_seed


def compute_fid(y_true, y_gen):
    """
    Compute the FID score between two tensors of true and predicted values using a simple model.

    Args:
    y_true (torch.Tensor): Ground truth images tensor of shape (batch_size, 1, 28, 28).
    y_gen (torch.Tensor): Generated images tensor of shape (batch_size, 1, 28, 28).

    Returns:
    float: The computed FID score.
    """

    # Define a simple feedforward neural network for feature extraction
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)),  # Output: (8, 14, 14)
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)),  # Output: (16, 7, 7)
        ('relu2', nn.ReLU()),
        ('flatten', nn.Flatten()),
        ('fc', nn.Linear(16 * 7 * 7, 10)),  # Arbitrary feature dimension size
        ('output', nn.ReLU())
    ]))

    # Define a function that will handle the engine's processing step
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            return batch

    # Create an evaluator engine
    evaluator = Engine(eval_step)

    # Initialize the FID metric with the model as the feature extractor
    metric = FID(num_features=10, feature_extractor=model)
    metric.attach(evaluator, "fid")

    # Run evaluation
    state = evaluator.run([[y_gen, y_true]])

    return state.metrics["fid"]


# # Example usage within another script after importing
# if __name__ == "__main__":
#     manual_seed(666)
#     y_true = torch.zeros(25, 1, 28, 28)  # Placeholder for real images
#     y_pred = torch.ones(25, 1, 28, 28)  # Placeholder for generated images
#     fid_score = compute_fid(y_true, y_pred)
#     print("FID Score:", fid_score)
