import torch
import torch.nn as nn

class ConsistancyDiscriminator(nn.Sequential):
    '''
    Estimate the consistency of the predicted malignant probability and BIRADs classes distirbution.
    The consistency is a float in [0, 1].
    So if to be used in loss, use (1 - consistency).
    '''
    def __init__(self, K):
        nn.Sequential.__init__(
            self, 
            nn.Linear(K + 1, 1, False), 
            nn.Tanh()
        )
        
    def forward(self, Pm, Pb):
        '''
        Pm: [N, 2]
        Pb: [N, K]
        O: [N, 1]. in (0, 1)
        '''
        x = torch.cat((Pm[:, 1:2], Pb), dim=-1)
        x = nn.Sequential.forward(self, x)
        # NOTE: use tanh + square here because sigmoid(0) != 0, 
        # but for the non-linear f(x) applied on the Linear output, f(0) should be 0.
        return 1 - torch.pow(x, 2)

    def loss(self, Pm, Pb, Y):
        '''
        Pm: [N, 2]
        Pb: [N, K]
        Y: [N, 1]
        '''
        return nn.functional.mse_loss(
            self.forward(Pm, Pb), Y
        )