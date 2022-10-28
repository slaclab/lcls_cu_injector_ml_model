import json

import torch
from botorch.models.transforms.input import Normalize
from torch.nn import Module


def create_nn_transformers(fname):
    """return normalize transformers from fname"""
    data = json.load(open(fname))

    transformers = []
    for ele in ["x", "y"]:
        transform = Normalize(len(data[f"{ele}_min"]))
        transform.ranges = 1 / torch.tensor(
            data[f"{ele}_scale"], dtype=torch.double
        ).unsqueeze(0)
        transform.mins = (
            -torch.tensor(data[f"{ele}_min"], dtype=torch.double).unsqueeze(0)
            * transform.ranges
        )

        transform.eval()
        transformers.append(transform)

    return transformers


class Surrogate(Module):
    def __init__(
        self,
        model: Module,
        model_input_list: list,
        model_output_list: list,
        model_input_transform: Normalize,
        model_output_transform: Normalize,
    ):
        """
        Surrogate model trained on simulation data
        - surrogate model passed to `model` is trained on model_input_transform(train_x)
        and model_output_transform(log(train_y))

        :param model:
        """

        super(Surrogate, self).__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()

        self.model_input_list = model_input_list
        self.model_output_list = model_output_list

        self.input_transform = model_input_transform
        self.input_transform.eval()
        self.output_transform = model_output_transform
        self.output_transform.eval()

    def forward(self, X, return_log=False):
        """input X in simulation units, makes prediction in simulation units"""
        X = self.input_transform(X)
        out = self.model(X)
        out = self.output_transform.untransform(out)

        if not return_log:
            out = torch.exp(out)

        return out
