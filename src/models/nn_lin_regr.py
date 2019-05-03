import torch


class NNLinRegr(torch.nn.Module):
    def __init__(self, name, d_in, *, d_out=1, dtype=torch.float, device='cpu'):
        super(NNLinRegr, self).__init__()
        self.dtype = dtype
        self.device = device
        self.name = name
        self.lin1 = torch.nn.Linear(d_in, d_out)

    def forward(self, x):
        z = self.lin1(x)
        return z



