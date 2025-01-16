import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self, input_dim, hidden_dims=[256, 256], activation_type="ReLU", backbone=None, num_img=1
    ):
        super(Discriminator, self).__init__()
        self.encoder = backbone
        self.num_img = num_img
        layers = []
        dims = [self.encoder.repr_dim * num_img + input_dim] + hidden_dims
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
        rgb = obs_dict["rgb"]
        actions = obs_dict["actions"]
        if self.num_img > 1:
            B, C, H, W = rgb.shape
            rgb = rgb.reshape(B, self.num_img, 3, H, W)
            rgb_features = [self.encoder(rgb[:, i, ...], flatten=True) for i in range(self.num_img)]
            rgb_features = torch.cat(rgb_features, dim=1)
        else:
            rgb_features = self.encoder(rgb, flatten=True)
        # Image encoding + proprio state + action
        x = torch.cat([rgb_features, state, actions], dim=1)
        return self.model(x)
