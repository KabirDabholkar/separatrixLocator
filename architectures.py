import torch
import torch.nn as nn
import torch.nn.functional as F
# from s4.models.s4.s4 import S4Block
from s4torch import S4Model



class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
            # nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)  # Skip connection


class DeepResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks):
        super(DeepResNet, self).__init__()
        # Add an input layer if input_dim < hidden_dim to map input to hidden_dim
        if input_dim < hidden_dim:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_layer = None

        # Stack the ResidualBlocks in sequence (operating on hidden_dim)
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # Add an output layer if output_dim is not equal to hidden_dim
        if output_dim != hidden_dim:
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_layer = None

    def forward(self, x):
        # If an input layer is defined, use it to transform x
        if self.input_layer:
            x = self.input_layer(x)
        # Pass the result through the stacked residual blocks
        x = self.residual_layers(x)
        # If an output layer is defined, map the hidden state to output_dim
        if self.output_layer:
            x = self.output_layer(x)
        return x



# Define a custom module to add a singleton dimension.
class Unsqueeze(nn.Module):
    def __init__(self, dim=-1):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

# class S4SequenceModel(nn.Module):
#     def __init__(self, d_model, **s4_kwargs):
#         super().__init__()
#         self.s4 = S4Block(
#             d_model,
#             transposed=False,
#             **s4_kwargs
#         )
#
#     def forward(self, x):
#         """
#         x: Tensor of shape (B, L)
#         Returns: Tensor of shape (B, L)
#         """
#         if x.ndim != 2:
#             raise ValueError(f"Expected input of shape (B, L), got {x.shape}")
#
#         # Reshape to (B, L, 1)
#         x = x.unsqueeze(-1)
#
#         # Pass through S4Block
#         y, _ = self.s4(x)  # output shape: (B, L, 1)
#
#         # Reshape back to (B, L)
#         return y.squeeze(-1)
#
#
# class S4FinalPredictor(nn.Module):
#     def __init__(self, input_size, output_size, d_model=1, **s4_kwargs):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#
#         # S4Block expects input shape (B, L, H)
#         self.s4 = S4Block(
#             d_model=d_model,
#             transposed=False,
#             # n_ssm=ssm_state_size,
#             **s4_kwargs
#         )
#
#         # Final linear layer projects the last S4 output to desired output size
#         self.proj = nn.Linear(d_model, output_size)
#
#     def forward(self, x):
#         """
#         x: Tensor of shape (B, input_size)
#         Returns: Tensor of shape (B, output_size)
#         """
#         B, L = x.shape
#         x = x.unsqueeze(-1)  # (B, L, 1)
#
#         y, _ = self.s4(x)    # y: (B, L, 1)
#
#         y_last = y[:, -1, :] # (B, 1)
#
#         out = self.proj(y_last)  # (B, output_size)
#
#         return out

if __name__ == "__main__":
    # model = S4Block(
    #     10
    # )
    # outputs = model(torch.zeros(10, 1, 20))
    # print(
    #     outputs[0].shape
    # )

    # model = S4SequenceModel(d_model=10)
    # model = S4FinalPredictor(input_size=1,output_size=4,d_model=10)
    # outputs = model(torch.zeros(10, 20))
    # print(
    #     outputs.shape
    # )

    N = 32
    d_input = 1
    d_model = 128
    n_classes = 10
    n_blocks = 3
    seq_len = 784

    u = torch.randn(3, seq_len, d_input)

    s4model = S4Model(
        d_input,
        d_model=d_model,
        d_output=n_classes,
        n_blocks=n_blocks,
        n=N,
        l_max=seq_len,
        collapse=True,  # average predictions over time prior to decoding
    )
    print(
        s4model(u).shape,
        u.shape
    )
    model = nn.Sequential(
        Unsqueeze(dim=-1),
        s4model,
    )
    model(u[...,0]).shape