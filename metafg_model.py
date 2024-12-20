from collections import OrderedDict

import torch
import torchvision
from torch import nn


def get_metafg_model():
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 11000)

    # Load the state dict
    state_dict = torch.load('/home/sunjiao/Downloads/model_best.pth.tar', map_location=torch.device('cpu'))[
        'state_dict']

    # Create a new state dict with keys modified
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        new_state_dict[name] = v

    # Load the new state dict
    model.load_state_dict(new_state_dict)
    return model