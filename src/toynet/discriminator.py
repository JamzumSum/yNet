import torch
import torch.nn as nn

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