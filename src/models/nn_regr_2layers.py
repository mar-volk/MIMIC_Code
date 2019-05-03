import torch


class NNLinRegr2Layers(torch.nn.Module):
    def __init__(self, name, d_in, *, d_h=200, d_out=1, p_drop=0.1, dtype=torch.float, device='cpu'):
        super(NNLinRegr2Layers, self).__init__()
        self.dtype = dtype
        self.device = device
        self.name = name

        self.dropout1 = torch.nn.modules.Dropout(p=p_drop)
        self.lin1 = torch.nn.Linear(d_in, d_h)
        self.activate1 = torch.nn.Sigmoid()

        self.dropout2 = torch.nn.modules.Dropout(p=p_drop)
        self.lin2 = torch.nn.Linear(d_h, d_out)

    def forward(self, x):
        z = self.dropout1(x)
        z = self.lin1(z)
        z = self.activate1(z)
        z = self.dropout2(z)
        z = self.lin2(z)
        return z


