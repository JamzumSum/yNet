import torch
import torch.nn as nn
import torch.nn.functional as F
from common.support import HasDiscriminator, MultiBranch
from common import freeze
from .lossbase import HasLoss, MultiTask


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

    def __loss__(self, pm, pb, real=True, clip=0.) -> tuple:
        r = {'cons': (c := self.forward(pm, pb))}
        y = int(real)
        loss = {'sd': (c.squeeze(1) - y).pow(2).clamp(min=clip)}
        return r, loss


def WithSD(ToyNet: type[MultiTask], *darg, **dkwarg):
    # BUG: inner-class cannot be serialized
    class DiscriminatorAssembler(ToyNet, HasDiscriminator, MultiBranch):
        def __init__(self, *args, **argv):
            ToyNet.__init__(self, *args, **argv)
            self.D = SimpleDiscriminator(self.K, *darg, **dkwarg)

        def branches(self):
            return (*super().branches, 'D')
            
        def branch_weight(self, weight_decay: dict):
            d = super().branch_weight(weight_decay)
            d['D'] = self.D.parameters()
            return d

        def __loss__(self, meta, *args, reduce=True, **kwargs) -> tuple:
            res, d = ToyNet.__loss__(meta, *args, reduce=False, **kwargs)

            # NOTE: this training flag is that of the main model.
            # Thus when ynet is training, we hope it generate real prob;
            # When ynet is not training (discriminator is training), we hope it expose them as fake.
            r, sd = self.D.__loss__(res['pm'], res['pb'], real=self.training)
            res |= r
            d |= sd

            if reduce: d = self.reduceLoss(d, meta['augindices'])
            return res, d

    return DiscriminatorAssembler
