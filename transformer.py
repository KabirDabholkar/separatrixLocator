import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        use_positional_encoding: bool = True,
        pooling: str = 'mean',  # Options: 'mean', 'max', 'attention'
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.pooling = pooling.lower()

        # Project each scalar input dimension to a d_model vector
        self.input_proj = nn.Linear(1, d_model)

        # Optional learnable positional encoding
        if use_positional_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(input_dim, d_model))
        else:
            self.pos_embedding = None

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional attention pooling
        if self.pooling == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softmax(dim=1)
            )

        # Output mapping
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        # x: [B, input_dim] → [B, input_dim, 1]
        x = x.unsqueeze(-1)

        # Project to token embeddings: [B, input_dim, d_model]
        x = self.input_proj(x)

        # Add positional encoding if used
        if self.pos_embedding is not None:
            x = x + self.pos_embedding

        # Transformer encoder
        x = self.transformer(x)  # [B, input_dim, d_model]

        # Pooling over input dimensions
        if self.pooling == 'mean':
            x_pooled = x.mean(dim=1)  # [B, d_model]
        elif self.pooling == 'max':
            x_pooled = x.max(dim=1).values
        elif self.pooling == 'attention':
            weights = self.attn_pool(x)  # [B, input_dim, 1]
            x_pooled = (x * weights).sum(dim=1)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling}")

        # Output: [B, output_dim]
        return self.output_head(x_pooled)

def main():
    # Settings
    batch_size = 8
    input_dim = 20
    output_dim = 7

    # Initialize model
    model = Transformer(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        use_positional_encoding=True,
        pooling='attention'  # Try 'max' or 'attention' as well
    )

    # Create a random batch of inputs
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    psi = model(x)

    # Output info
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {psi.shape}")
    print(f"Sample output: {psi[0]}")

if __name__ == "__main__":
    main()