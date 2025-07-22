import torch
import torch.nn as nn
from torch.autograd import Function


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.residual = (
            nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)
        )

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.drop(out)
        return out + self.residual(x)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)




class CLSBiasAdderMLP(nn.Module):
    def __init__(self, cls_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(cls_dim, cls_dim//4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(cls_dim//4, cls_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, cls_embedding, apply_bias: bool = True):
        if not apply_bias:
            return cls_embedding  # identity
        
        print(f"input type: {cls_embedding.dtype}")
        bias = self.fc1(cls_embedding.float())
        bias = self.relu(bias)
        bias = self.dropout(bias)
        bias = self.fc2(bias)
        return cls_embedding + bias

    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            

class CLSDiscriminatorMLP(nn.Module):
    def __init__(self, cls_dim: int, num_clusters: int, hidden_dim: int = 512, dropout: float = 0.1):
        """
        Args:
            cls_dim (int): Dimensionality of the CLS embedding from the vision encoder.
            num_clusters (int): Number of output clusters (i.e., classes).
            hidden_dim (int): Size of the hidden layer.
            dropout (float): Dropout rate between layers.
        """
        super(CLSDiscriminatorMLP, self).__init__()
        self.num_clusters = num_clusters
        self.net = nn.Sequential(
            nn.Linear(cls_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_clusters if num_clusters > 2 else 1)
        )
        self.apply(init_weights)
        

    def forward(self, cls_embedding, label=None):
        """
        Args:
            cls_embedding (Tensor): shape (B, cls_dim)
        Returns:
            logits (Tensor): shape (B, num_clusters)
        """
        logits = self.net(cls_embedding)
        if label is not None:
            loss = get_discriminator_loss(self.num_clusters)(logits, label)
            return logits, loss
        return logits





def get_discriminator_loss(num_clusters):
    if num_clusters == 1:
        # BCEWithLogitsLoss expects float labels of shape (B, 1)
        return nn.BCEWithLogitsLoss()
    elif num_clusters == 2:
        # BCEWithLogitsLoss for binary, but you may want to use CrossEntropyLoss if your labels are class indices (0/1)
        # If your labels are one-hot or float, use BCEWithLogitsLoss
        return nn.BCEWithLogitsLoss()
    else:
        # CrossEntropyLoss expects class indices (LongTensor) of shape (B,)
        return nn.CrossEntropyLoss()