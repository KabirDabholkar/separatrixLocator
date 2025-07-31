import torch
import torch.nn as nn
import torch.nn.functional as F
# from s4.models.s4.s4 import S4Block
from s4torch import S4Model

class PadLayer(nn.Module):
    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x):
        if x.shape[-1] < self.target_dim:
            # Pad with zeros to reach target dimension
            pad_size = self.target_dim - x.shape[-1]
            return F.pad(x, (0, pad_size), "constant", 0)
        return x

class TruncateLayer(nn.Module):
    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x):
        if x.shape[-1] > self.target_dim:
            # Truncate to target dimension
            return x[..., :self.target_dim]
        return x


class AdditiveModel(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList(models)  # Use ModuleList to register models

    def forward(self, x):
        return sum(model(x) for model in self.models)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, nonlin=nn.Tanh()):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nonlin
            # nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)  # Skip connection


class DeepResNetVariableWidth(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, nonlin=nn.Tanh()):
        """
        A ResNet architecture that can handle both increasing and decreasing widths.
        For increasing widths: uses padding
        For decreasing widths: uses truncation
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden dimensions for each residual block
            output_dim: Dimension of output features
            nonlin: Nonlinearity to use in residual blocks
        """
        super().__init__()
        
        # Input layer maps to first hidden dimension
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # Create residual blocks with variable widths
        self.residual_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            # Main residual block path - single linear layer
            self.residual_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nonlin
                )
            )
            
            # Skip connection handling
            if hidden_dims[i] < hidden_dims[i+1]:
                # If next layer is wider, pad with zeros
                self.residual_layers.append(PadLayer(hidden_dims[i+1] - hidden_dims[i]))
            elif hidden_dims[i] > hidden_dims[i+1]:
                # If next layer is narrower, truncate
                self.residual_layers.append(TruncateLayer(hidden_dims[i+1]))
            else:
                # If same width, use identity
                self.residual_layers.append(nn.Identity())
        
        # Output layer maps from last hidden dimension to output dimension
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for i in range(0, len(self.residual_layers), 2):
            # Get residual block and skip connection
            residual_block = self.residual_layers[i]
            skip_connection = self.residual_layers[i+1]
            
            # Apply residual block
            out = residual_block(x)
            
            # Apply skip connection (padding, truncation, or identity)
            x_skip = skip_connection(x)
            
            x = x_skip + out
            
        return self.output_layer(x)




class DeepResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks, nonlin = nn.Tanh()):
        super(DeepResNet, self).__init__()
        # Add an input layer if input_dim < hidden_dim to map input to hidden_dim
        if input_dim < hidden_dim:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_layer = None

        # Stack the ResidualBlocks in sequence (operating on hidden_dim)
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(hidden_dim, nonlin=nonlin) for _ in range(num_blocks)]
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

    # N = 32
    # d_input = 1
    # d_model = 128
    # n_classes = 10
    # n_blocks = 3
    # seq_len = 784

    # u = torch.randn(3, seq_len, d_input)

    # s4model = S4Model(
    #     d_input,
    #     d_model=d_model,
    #     d_output=n_classes,
    #     n_blocks=n_blocks,
    #     n=N,
    #     l_max=seq_len,
    #     collapse=True,  # average predictions over time prior to decoding
    # )
    # print(
    #     s4model(u).shape,
    #     u.shape
    # )
    # model = nn.Sequential(
    #     Unsqueeze(dim=-1),
    #     s4model,
    # )
    # model(u[...,0]).shape

    # model = AdditiveModel(nn.Linear(10,2),nn.Linear(10,2))

    # import torch
    # import torch.nn as nn
    # import torch.optim as optim
    #
    # # Sample data
    # x_train = torch.randn(100, 10)  # 100 samples, 10 features
    # y_train = torch.randint(0, 2, (100, 2)).float()  # 100 samples, 2 output classes
    #
    # # Define loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # # Training loop
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     model.train()
    #     optimizer.zero_grad()  # Clear gradients
    #     outputs = model(x_train)  # Forward pass
    #     loss = criterion(outputs, y_train)  # Compute loss
    #     loss.backward()  # Backward pass
    #     optimizer.step()  # Update weights
    #
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Create a DeepResNet model
    input_dim = 10
    hidden_dims = [64, 32, 16, 8]
    output_dim = 2
    num_blocks = 3
    model = DeepResNetVariableWidth(input_dim, hidden_dims, output_dim)

    # Generate some sample data
    x = torch.randn(5, input_dim)  # 5 samples, 10 features
    y = torch.randint(0, 2, (5, output_dim)).float()  # 5 samples, 2 output classes

    # Forward pass
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Sample output:", output[0])

    # # Define loss and optimizer
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # Training loop
    # num_epochs = 3
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     outputs = model(x)
    #     loss = criterion(outputs, y)
    #     loss.backward()
    #     optimizer.step()
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')