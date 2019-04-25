import torch


class NNRegr(torch.nn.Module):
    def __init__(self, name, d_in, d_m, d_out, dtype=torch.float, device='cpu'):
        super(NNRegr, self).__init__()
        self.dtype = dtype
        self.device = device
        self.name = name
        self.lin1 = torch.nn.Linear(d_in, d_m)
        self.lin2 = torch.nn.Linear(d_m, d_out)

    def forward(self, x):
        z = self.lin1(x)
        z = self.lin2(z)
        return z



