import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils import freeze

class ConsistancyDiscriminator(nn.Sequential):
    '''
    Estimate the consistency of the predicted malignant probability and BIRADs classes distirbution.
    The consistency is a float in [0, 1].
    So if to be used in loss, use (1 - consistency).
    '''
    def __init__(self, K, hs=64):
        nn.Sequential.__init__(
            self, 
            nn.Linear(K + 2, hs), 
            nn.Tanh(),
            nn.Linear(hs, hs), 
            nn.PReLU(),
            nn.Linear(hs, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, Pm, Pb):
        '''
        Pm: [N, 2]
        Pb: [N, K]
        O: [N, 1]. in (0, 1)
        '''
        x = torch.cat((Pm, Pb), dim=-1)
        return nn.Sequential.forward(self, x)

    def loss(self, Pm, Pb, Y):
        '''
        Pm: [N, 2]
        Pb: [N, K]
        Y: [N, 1]
        '''
        return nn.functional.mse_loss(
            self.forward(Pm, Pb), Y
        )

def WithCD(ToyNet, *darg, **dkwarg):
    sp = ToyNet.support
    if 'discriminator' in sp: raise ValueError(str(ToyNet), 'already supports discriminator.')
    sp = ('discriminator', *sp)

    class DiscriminatorAssembler(ToyNet):
        support = sp
        def __init__(self, *args, **argv):
            ToyNet.__init__(self, *args, **argv)
            self.D = ConsistancyDiscriminator(self.K, *darg, **dkwarg)
        
        def discriminatorParameters(self):
            return self.D.parameters()

        def discriminatorLoss(self, X, Ym, Yb, piter=0.):
            N = Ym.shape[0]
            with torch.no_grad():
                _, _, Pm, Pb = self.forward(X)
            loss1 = self.D.loss(Pm, Pb, torch.zeros(N, 1).to(X.device))
            loss2 = self.D.loss(
                F.one_hot(Ym, num_classes=2).type_as(Pm), 
                F.one_hot(Yb, num_classes=self.K).type_as(Pb), 
                torch.ones(N, 1).to(X.device)
            )
            loss2 = freeze(loss2, (1 - loss2.detach()) ** 2)    # like focal
            return (loss1 + loss2) / 2

        def _loss(self, *args, **argv):
            res, zipM, zipB, zipC = ToyNet._loss(self, *args, **argv)
            if zipC is None: zipC = []
            _, _, Pm, Pb = res
            consistency = self.D.forward(Pm, Pb).mean()
            return res, zipM, zipB, [consistency] + zipC

        def lossWithResult(self, *args, **argv):
            '''
            return: Original result, M-branch losses, B-branch losses, consistency.
            '''
            res, loss, summary = ToyNet.lossWithResult(self, *args, **argv)
            # But ToyNetV1 can constrain between the probability distributions Pm & Pb :D
            consistency = res[3][0]
            summary['interbranch/consistency'] = consistency.detach()
            return res, loss + (1 - consistency), summary
    return DiscriminatorAssembler