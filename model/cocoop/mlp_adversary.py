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


class AdversarialMLP(nn.Module):
    def __init__(self, input_dim, opt, output_dim=1, use_bias_ctx=False, n_ctx=4):
        super().__init__()
        hidden_dims = opt.hidden_structure

        layers = []

        if use_bias_ctx:
            # Add a bias context layer if specified
            layers.append(nn.Linear(512*2, hidden_dims[0]))
            layers.append(nn.ReLU())
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
                layers.append(ResidualBlock(in_dim, out_dim))
        else:
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
                layers.append(ResidualBlock(in_dim, out_dim))

        # Final output layer with configurable output_dim
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

        self.model.apply(self.init_weights.__get__(self, AdversarialMLP))

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        out = self.forward(x)
        return torch.sigmoid(out).squeeze(-1) if out.shape[-1] == 1 else torch.sigmoid(out)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
