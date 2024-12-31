import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self, input_dim, hidden_dims=[256, 256], activation_type="ReLU", backbone=None
    ):
        super(Discriminator, self).__init__()
        self.encoder = backbone
        layers = []
        dims = [self.encoder.repr_dim + input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if activation_type == "ReLU":
                layers.append(nn.ReLU())
            elif activation_type == "LeakyReLU":
                layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, obs_dict):
        state = obs_dict["state"]
        images = obs_dict["rgb"]
        actions = obs_dict["actions"]
        image_features = self.encoder(images)
        # Image encoding + proprio state + action
        x = torch.cat([image_features.flatten(1), state, actions], dim=1)
        return self.model(x)
