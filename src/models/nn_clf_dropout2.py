import torch

# Neural Network
class NNClfDropout2(torch.nn.Module):
    def __init__(self, name, d_in, d_out, p_drop=0.3, dtype=torch.float, device='cpu'):
        super(NNClfDropout2, self).__init__()
        self.dtype = dtype
        self.device = device
        self.name = name
        self.p_drop = p_drop

        self.dropout1 = torch.nn.modules.Dropout(p=p_drop)
        self.lin1 = torch.nn.Linear(d_in, 30)
        self.activate1 = torch.nn.ReLU()

        self.dropout2 = torch.nn.modules.Dropout(p=p_drop)
        self.lin2 = torch.nn.Linear(30, d_out)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        z = self.dropout1(x)
        z = self.lin1(z)
        z = self.activate1(z)
        z = self.dropout2(z)
        z = self.lin2(z)
        return z

    def predict_proba(self, x):
        z = self.forward(x)
        prob = self.softmax(z).data
        return (prob)

    def predict(self, x):
        z = self.forward(x)
        pred = torch.max(z.data, 1)[1]
        return (pred)
