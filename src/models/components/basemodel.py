## Define a MLP model with N layers

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for i in range(self.n_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.Dropout(0.5))
            # self.layers.append(nn.BatchNorm1d(self.hidden_dim))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

# Define a Model with a embbeding layer and a MLP

class ClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim, embbeding_dim, hidden_out, hidden_dim=128, n_layers=2):
        super(ClassificationModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embbeding_dim = embbeding_dim
        self.embbeding_out = hidden_out
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embbeding_layer = nn.Embedding(self.embbeding_dim, self.embbeding_out)
        self.mlp = MLP(self.input_dim * self.embbeding_out, self.output_dim, self.hidden_dim, self.n_layers)

    def forward(self, x):
        x = self.embbeding_layer(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        ## classification
        x = F.softmax(x, dim=1)
        return x

if __name__ == "__main__":
    _ = ClassificationModel(10, 2, 10, 10)