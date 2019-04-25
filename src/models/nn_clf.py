import torch

# Neural Network
class NNClf(torch.nn.Module):
    def __init__(self, name, d_in, d_out, dtype=torch.float, device='cpu'):
        super(NNClf, self).__init__()
        self.dtype = dtype
        self.device = device
        self.name = name
        self.lin1 = torch.nn.Linear(d_in, 30)
        self.activate1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(30, d_out)
        self.softmax = torch.nn.Softmax(dim=1)   # ()

    def forward(self, x):
        z = self.lin1(x)
        z = self.activate1(z)
        z = self.lin2(z)
        return z

    def predict_proba(self, x):
        z = self.forward(x)
        print('In NN_clf, z.shape')
        print(z.shape)
        prob = self.softmax(z).data
        return (prob)

    def predict(self, x):
        z = self.forward(x)
        pred = torch.max(z.data, 1)[1]
        return (pred)



