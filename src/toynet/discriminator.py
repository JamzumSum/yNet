import torch
import torch.nn as nn
import torch.nn.functional as F
from common import freeze
from .lossbase import HasLoss


def rand_smooth_label(target: torch.Tensor, smooth=0.1, K=-1) -> torch.Tensor:
    mask = F.one_hot(target, num_classes=K).float()
    rnd = torch.rand_like(target)
    return mask * rnd * (1 - smooth) + (1 - mask) * rnd * smooth


class ManualDiscriminator(nn.Module, HasLoss):
    def __init__(self, K, p=1) -> None:
        super().__init__()
        assert K == 3
        self.p = p

        weight = [0.02, 0.5, 0.95]
        self.register_buffer('weight', torch.Tensor(weight))

    def forward(self, Pm, Pb):
        """calculate consistency of pm and pb using manually given weights.

        Args:
            Pm (Tensor): [N, 2]
            Pb (Tensor): [N, K]

        Returns:
            Tensor: [N, 1]. in (0, 1)
        """
        dist = (Pb * self.weight).sum(dim=-1) - Pm[:, -1] # [N]
        dist = dist.unsqueeze(-1)                         # [N, 1]
        return torch.norm(1 - dist, self.p, dim=1, keepdim=True)

    def __loss__(self, pm, pb, clip=0.9) -> tuple:
        r = {'cons': (c := self.forward(pm, pb))}
        loss = {'sd': (1 - c.squeeze(1)).clamp_min(clip)}
        return r, loss


class SimpleDiscriminator(nn.Sequential, HasLoss):
    '''
    Estimate the consistency of the predicted malignant probability and BIRADs classes distirbution.
    The consistency is a float in [0, 1].
    '''
    def __init__(self, K, hs=64):
        nn.Sequential.__init__(
            self, nn.Linear(K + 2, hs), nn.Tanh(), nn.Linear(hs, hs), nn.PReLU(),
            nn.Linear(hs, 1), nn.Sigmoid()
        )

    def forward(self, Pm, Pb):
        """calculate consistency of pm and pb.

        Args:
            Pm (Tensor): [N, 2]
            Pb (Tensor): [N, K]

        Returns:
            Tensor: [N, 1]. in (0, 1)
        """
        x = torch.cat((Pm, Pb), dim=-1)
        return nn.Sequential.forward(self, x)

    def __loss__(self, pm, pb, real=True, clip=0., reduce=False) -> tuple:
        r = {'cons': (c := self.forward(pm, pb))}
        y = int(real)
        loss = {'sd': (y - c.squeeze(1)).pow(2).clamp_min(clip)}
        return r, loss
