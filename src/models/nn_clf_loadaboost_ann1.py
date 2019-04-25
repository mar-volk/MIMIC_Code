import torch

# Neural Network as proposed in
# Huang et al. "LoAdaBoost:Loss-Based AdaBoost Federated Machine Learning on medical Data"
# https://github.com/liasktao/LoAdaBoost/blob/master/IID_average_evaluation.py
class NNClfLoAdaBoost_ann1(torch.nn.Module):
    def __init__(self, name, d_in, d_out, p_drop=0.5, dtype=torch.float, device='cpu'):
        super(NNClfLoAdaBoost_ann1, self).__init__()
        self.dtype = dtype
        self.device = device
        self.name = name
        self.p_drop = p_drop

        self.lin1 = torch.nn.Linear(d_in, 20)
        self.activate1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.modules.Dropout(p=p_drop)

        self.lin2 = torch.nn.Linear(20, 10)
        self.activate2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.modules.Dropout(p=p_drop)

        self.lin3 = torch.nn.Linear(10, 5)
        self.activate3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.modules.Dropout(p=p_drop)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        z = self.lin1(x)
        z = self.activate1(z)
        z = self.dropout1(z)

        z = self.lin2(z)
        z = self.activate2(z)
        z = self.dropout2(z)

        z = self.lin3(z)
        z = self.activate3(z)
        z = self.dropout3(z)
        return z

    def predict_proba(self, x):
        z = self.forward(x)
        prob = self.sigmoid(z).data
        return prob

    def predict(self, x):
        z = self.forward(x)
        pred = torch.max(z.data, 1)[1]
        return pred
