import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


def gradient_penalty(
    learner_sa: torch.Tensor,
    expert_sa: torch.Tensor,
    f: nn.Module,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Calculates the gradient penalty for the given learner and expert state-action tensors.

    Args:
        learner_sa (torch.Tensor): The state-action tensor from the learner.
        expert_sa (torch.Tensor): The state-action tensor from the expert.
        f (nn.Module): The discriminator network.
        device (str, optional): The device to use. Defaults to "cuda".

    Returns:
        torch.Tensor: The gradient penalty.
    """
    batch_size = expert_sa.size()[0]

    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(expert_sa)

    # interpolated = {}
    # for key in expert_sa.keys():
    #     expand_shape = (batch_size,) + (1,) * (expert_sa[key].ndim - 1)
    #     # _alpha = alpha.expand_as(expert_sa[key])
    #     _alpha = alpha.view(expand_shape)
    #     interpolated[key] = _alpha * expert_sa[key].data + (1 - _alpha) * learner_sa[key].data
    #     interpolated[key] = Variable(interpolated[key], requires_grad=True).to(device)

    interpolated = alpha * expert_sa.data + (1 - alpha) * learner_sa.data
    interpolated = Variable(interpolated, requires_grad=True).to(device)

    f_interpolated = f(interpolated).to(device)

    gradients = torch_grad(
        outputs=f_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(f_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    gradients = gradients.view(batch_size, -1)

    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    # 2 * |f'(x_0)|
    return ((gradients_norm - 0.4) ** 2).mean()


class Discriminator(nn.Module):
    def __init__(
        self, input_dim, hidden_dims=[256, 256], activation_type="ReLU", backbone=None, num_img=1, action_free=False,
    ):
        super(Discriminator, self).__init__()
        self.encoder = backbone
        self.num_img = num_img
        self.action_free = action_free
        layers = []
        dims = [(0 if self.encoder is None else self.encoder.repr_dim) * num_img + input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if activation_type == "ReLU":
                layers.append(nn.ReLU())
            elif activation_type == "LeakyReLU":
                layers.append(nn.LeakyReLU())
            elif activation_type == "SiLU":
                layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def preprocess_data(self, obs_dict):
        if "rgb" in obs_dict:
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
            if self.action_free:
                x = torch.cat([rgb_features, state], dim=1)
            else:
                # Image encoding + proprio state + action
                x = torch.cat([rgb_features, state, actions], dim=1)
        else:
            if self.action_free:
                x = obs_dict["state"]
            else:
                x = torch.cat([obs_dict["state"], obs_dict["actions"]], dim=1)

        return x

    def forward(self, obs_dict):
        x = self.preprocess_data(obs_dict)
        return self.model(x)
