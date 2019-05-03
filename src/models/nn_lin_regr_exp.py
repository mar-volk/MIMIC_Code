import torch


class NNLinRegrExp(torch.nn.Module):
    def __init__(self, name, d_in, *, d_out=1, dtype=torch.float, device='cpu'):
        super(NNLinRegrExp, self).__init__()
        self.dtype = dtype
        self.device = device
        self.name = name
        self.lin1 = torch.nn.Linear(d_in, d_out)

    def forward(self, x):
        z = self.lin1(x)
        z = torch.exp(z)
        return z



