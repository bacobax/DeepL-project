import torch
import torch.nn as nn
from torch.autograd import Function


class ResidualBlock(nn.Module):
    """
    A residual block with a linear layer, layer normalization, ReLU activation, dropout, and optional identity or projection shortcut.
    """
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
        """
        Forward pass through the residual block with normalization, activation, dropout, and residual connection.
        """
        out = self.linear(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.drop(out)
        return out + self.residual(x)


class GradientReversalFunction(Function):
    """
    Implements a gradient reversal layer as a custom autograd function, useful in adversarial training.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass that returns the input as-is and stores the lambda factor for use in the backward pass.
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass that reverses the gradient by multiplying it by -lambda.
        """
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    A wrapper for the GradientReversalFunction to integrate it into a standard nn.Module.
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        """
        Applies the gradient reversal function with the stored lambda parameter.
        """
        return GradientReversalFunction.apply(x, self.lambda_)


class AdversarialMLP(nn.Module):
    """
    A multi-layer perceptron with optional bias context support, designed for adversarial learning.
    Uses residual blocks for intermediate layers and configurable final output dimension.
    """
    def __init__(self, input_dim, opt, output_dim=1, use_bias_ctx=False, n_ctx=4):
        """
        Initializes the adversarial MLP with given structure and optional bias context input.
        Builds the network using residual blocks and applies Xavier initialization to weights.
        """
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
        """
        Forward pass through the adversarial MLP model.
        """
        return self.model(x)

    def predict(self, x):
        """
        Performs a forward pass and applies a sigmoid activation to the output.
        Squeezes output if it's a single dimension.
        """
        out = self.forward(x)
        return torch.sigmoid(out).squeeze(-1) if out.shape[-1] == 1 else torch.sigmoid(out)

    def init_weights(self, m):
        """
        Initializes weights of linear layers using Xavier uniform distribution and biases to zero.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
