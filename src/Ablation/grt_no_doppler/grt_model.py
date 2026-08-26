"""GRT-Small Model - from official codebase.

This implementation directly copies necessary modules from the official GRT codebase
(grt/deepradar/modules).
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Sequence
import numpy as np
from einops import rearrange

# ============================================================================
# Official GRT Modules (copied from grt/deepradar/modules/*.py)
# ============================================================================


class PatchMerge(nn.Module):
    """Merge patches with normalization and nominally reduced projection.

    From: grt/deepradar/modules/patch.py
    """

    def __init__(
        self, d_in: int, d_out: int, scale: Sequence[int] = [], norm: bool = True
    ) -> None:
        super().__init__()

        self.scale = scale
        d_merge = d_in * int(np.prod(scale))
        self.linear = nn.Linear(d_merge, d_out, bias=False)
        self.norm = nn.LayerNorm(d_merge) if norm else None

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        """Perform patch merging."""
        n, *t, c = x.shape
        dims = sum(([d // s, s] for d, s in zip(t, self.scale)), start=[n])
        order = (
            [0]
            + [2 * i + 1 for i in range(len(self.scale))]
            + [2 * i + 2 for i in range(len(self.scale))]
            + [-1]
        )
        t2 = [d // s for d, s in zip(t, self.scale)]
        return x.reshape(dims + [c]).permute(order).reshape(n, *t2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge and project."""
        merged = self._merge(x)
        if self.norm is not None:
            merged = self.norm(merged)
        return self.linear(merged)


class Sinusoid(nn.Module):
    """Centered N-dimensional sinusoidal positional embedding.

    From: grt/deepradar/modules/position.py
    """

    def __init__(
        self,
        scale: Optional[Sequence[float]] = None,
        global_scale: float = 1.0,
        coef: float = 10000.0,
    ) -> None:
        super().__init__()
        if scale is None:
            self.scale = [global_scale]
        else:
            self.scale = [s * global_scale for s in scale]
        self.coef = coef

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sinusoidal embedding."""
        # w = coef ** (-i / c)
        nd = len(x.shape) - 2
        c = x.shape[-1] // 2 // nd
        i = torch.arange(c, device=x.device)
        w = self.coef ** (-i / c)

        start_dim = 0
        for axis, (d, scale) in enumerate(zip(x.shape[1:-1], self.scale * nd)):
            # t = scale * (j - d/2) / (d/2) = scale * (2j / d - 1)
            t = scale * (2 * (torch.arange(d, device=x.device) + 0.5) / d - 1)
            wt = t[:, None] * w[None, :]

            p_slice = [None] * (len(x.shape) - 1) + [slice(None)]
            p_slice[axis + 1] = slice(None)

            # pos[2 * i] = sin(w * t)
            x_sin_slice = [slice(None)] * len(x.shape)
            x_sin_slice[-1] = slice(start_dim, start_dim + c * 2, 2)
            x_sin_slice = tuple(x_sin_slice)
            p_slice_tuple = tuple(p_slice)
            x[x_sin_slice] = x[x_sin_slice] + torch.sin(wt)[p_slice_tuple]

            # pos[2 * i + 1] = cos(w * t)
            x_cos_slice = [slice(None)] * len(x.shape)
            x_cos_slice[-1] = slice(start_dim + 1, start_dim + c * 2 + 1, 2)
            x_cos_slice = tuple(x_cos_slice)
            x[x_cos_slice] = x[x_cos_slice] + torch.cos(wt)[p_slice_tuple]

            start_dim += c * 2

        return x


class Readout(nn.Module):
    """Add readout token (concatenating along the spatial axis).

    From: grt/deepradar/modules/position.py
    """

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.readout = nn.Parameter(data=torch.normal(0, 0.02, (d_model,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate readout token."""
        readout = torch.tile(self.readout[None, None, :], (x.shape[0], 1, 1))
        return torch.concatenate((x, readout), dim=1)


def transformer_mlp(
    d_model: int = 512,
    d_feedforward: int = 2048,
    activation: str = "GELU",
    dropout: float = 0.0,
    eps: float = 1e-5,
) -> nn.Module:
    """Create transformer MLP.

    From: grt/deepradar/modules/transformer.py
    """
    return nn.Sequential(
        nn.LayerNorm(d_model, eps=eps, bias=True),
        nn.Linear(d_model, d_feedforward, bias=True),
        getattr(nn, activation)(),
        nn.Dropout(dropout),
        nn.Linear(d_feedforward, d_model, bias=True),
        nn.Dropout(dropout),
    )


class TransformerLayer(nn.Module):
    """Single transformer (encoder) layer.

    Uses PyTorch's naming convention to match checkpoint:
    - self_attn (not attn)
    - linear1, linear2 (not feedforward.0, feedforward.4)
    - norm1, norm2 (for attention and feedforward)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        d_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "GELU",
    ) -> None:
        super().__init__()

        # Attention with PyTorch naming
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, bias=True, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # Feedforward with PyTorch naming
        self.linear1 = nn.Linear(d_model, d_feedforward, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model, bias=True)
        self.dropout2 = nn.Dropout(dropout)

        # Norms
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, bias=True)

        # Activation
        self.activation = getattr(nn, activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer with pre-norm (norm_first=True style)."""
        # Self attention block
        x2 = self.norm1(x)
        x2 = self.self_attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.dropout1(x2)

        # Feedforward block
        x2 = self.norm2(x)
        x2 = self.linear1(x2)
        x2 = self.activation(x2)
        x2 = self.dropout(x2)
        x2 = self.linear2(x2)
        x = x + self.dropout2(x2)

        return x


class TransformerDecoder(nn.Module):
    """Single transformer (decoder) layer.

    Uses PyTorch's naming convention to match checkpoint:
    - self_attn, multihead_attn (not attn, attn2)
    - linear1, linear2 (not feedforward.0, feedforward.4)
    - norm1, norm2, norm3 (for self-attn, cross-attn, and feedforward)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        d_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "GELU",
    ) -> None:
        super().__init__()

        # Self attention with PyTorch naming
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, bias=True, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # Cross attention with PyTorch naming (multihead_attn, not attn2)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, bias=True, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)

        # Feedforward with PyTorch naming
        self.linear1 = nn.Linear(d_model, d_feedforward, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model, bias=True)
        self.dropout3 = nn.Dropout(dropout)

        # Norms (note: norm2 is for cross-attention)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5, bias=True)

        # Activation
        self.activation = getattr(nn, activation)()

    def forward(self, x: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
        """Apply transformer decoder with pre-norm."""
        # Self attention block
        x2 = self.norm1(x)
        x2 = self.self_attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.dropout1(x2)

        # Cross attention block
        x2 = self.norm2(x)
        x2 = self.multihead_attn(x2, x_enc, x_enc, need_weights=False)[0]
        x = x + self.dropout2(x2)

        # Feedforward block
        x2 = self.norm3(x)
        x2 = self.linear1(x2)
        x2 = self.activation(x2)
        x2 = self.dropout(x2)
        x2 = self.linear2(x2)
        x = x + self.dropout3(x2)

        return x


class BasisChange(nn.Module):
    """Create "change-of-basis" query.

    From: grt/deepradar/modules/transformer.py
    """

    def __init__(
        self,
        shape: Sequence[int] = [],
        flatten: bool = True,
        scale: Optional[Sequence[float]] = None,
        global_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.pos = Sinusoid(scale=scale, global_scale=global_scale)
        self.shape = shape
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply change of basis."""
        idxs = tuple([slice(None)] + [None] * len(self.shape) + [slice(None)])
        query = self.pos(torch.tile(x[idxs], (1, *self.shape, 1)))

        if self.flatten:
            query = query.reshape(x.shape[0], -1, x.shape[-1])
        return query


class Unpatch(nn.Module):
    """Unpatch data.

    Args:
        output_size: output 2D shape.
        features: number of input features; should be `>= size * size`.
        size: patch size as (width, height, channels).
    """

    def __init__(
        self,
        output_size: Sequence[int],
        features: int = 512,
        size: Sequence[int] = (16, 16),
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(features, output_size[-1] * int(np.prod(size)))
        self.size = size
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform 2D unpatching.

        Operates in batch-spatial-feature order; spatial axes are flattened on
        the input, and unflattened in the output.
        """
        embedding = self.linear(x)

        if len(self.size) == 2:
            return rearrange(
                embedding,
                "n (x1 x2) (s1 s2 c) -> n (x1 s1) (x2 s2) c",
                x1=self.output_size[0] // self.size[0],
                x2=self.output_size[1] // self.size[1],
                s1=self.size[0],
                s2=self.size[1],
                c=self.output_size[-1],
            )
        elif len(self.size) == 3:
            return rearrange(
                embedding,
                "n (x1 x2 x3) (s1 s2 s3 c) -> n (x1 s1) (x2 s2) (x3 s3) c",
                x1=self.output_size[0] // self.size[0],
                x2=self.output_size[1] // self.size[1],
                x3=self.output_size[2] // self.size[2],
                s1=self.size[0],
                s2=self.size[1],
                s3=self.size[2],
                c=self.output_size[-1],
            )
        else:
            raise ValueError("Unpatch is only implemented for 2D and 3D tensors.")


# ============================================================================
# GRT Model Components
# ============================================================================


class GRTEncoder(nn.Module):
    """GRT Transformer Encoder matching official implementation."""

    def __init__(
        self,
        layers: int = 4,
        dim: int = 512,
        ff_ratio: float = 4.0,
        head_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "GELU",
        patch: list[int] = [2, 8, 2, 4],
        pos_scale: list[float] = [1.0, 1.0, 1.0, 1.0],
        global_scale: float = 16.0,
        input_channels: int = 2,
        positions: Literal["flat", "nd"] = "nd",
    ):
        super().__init__()

        # Patch embedding
        self.patch = PatchMerge(d_in=input_channels, d_out=dim, scale=patch, norm=False)

        # Position embedding
        self.positions = positions
        self.pos = Sinusoid(scale=pos_scale, global_scale=global_scale)

        # Readout token
        self.readout = Readout(d_model=dim)

        # Encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_feedforward=int(ff_ratio * dim),
                    d_model=dim,
                    n_head=dim // head_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Patch embedding
        embedded = self.patch(x)

        # Apply positional encoding
        if self.positions == "nd":
            embedded = self.pos(embedded)

        # Flatten spatial dimensions
        flat = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1])

        # Apply flat positional encoding if needed
        if self.positions == "flat":
            flat = self.pos(flat)

        # Add readout token
        x = self.readout(flat)

        # Apply encoder layers
        for layer in self.layers:
            x = layer(x)

        return x


class GRTDecoder3D(nn.Module):
    """GRT 3D Transformer Decoder matching official implementation."""

    def __init__(
        self,
        key: str = "map",
        layers: int = 4,
        dim: int = 512,
        ff_ratio: float = 4.0,
        head_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "GELU",
        shape: list[int] = [64, 128, 64],
        pos_scale: list[float] = [1.0, 1.0, 1.0],
        global_scale: float = 16.0,
        patch: list[int] = [8, 8, 8],
        out_dim: int = 0,
        positions: Literal["flat", "nd"] = "nd",
        mode: Literal["last", "pool"] = "last",
    ):
        super().__init__()

        self.key = key
        self.out_dim = out_dim
        self.mode = mode

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoder(
                    d_feedforward=int(ff_ratio * dim),
                    d_model=dim,
                    n_head=dim // head_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(layers)
            ]
        )

        # Query generation with position encoding
        query_shape = [s // p for s, p in zip(shape, patch)]
        if positions == "flat":
            query_shape = [int(np.prod(query_shape))]

        self.query = BasisChange(
            shape=query_shape, scale=pos_scale, global_scale=global_scale, flatten=True
        )

        # Unpatch to reconstruct output
        self.unpatch = Unpatch(
            output_size=(*shape, max(1, self.out_dim)), features=dim, size=patch
        )

    def forward(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass."""
        # Extract readout token or pool
        if self.mode == "last":
            x = encoded[:, -1, :]
        else:
            x = torch.mean(encoded, dim=1)

        # Generate query with positional encoding
        x = self.query(x)

        # Encoded features without readout token
        enc = encoded[:, :-1, :]

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, enc)

        # Unpatch to 3D output
        out = self.unpatch(x)

        # Squeeze channel dimension if binary output
        if self.out_dim == 0:
            out = out[..., 0]

        return {self.key: out}


# ============================================================================
# Complete GRT-Small Model
# ============================================================================


class GRTSmall(nn.Module):
    """GRT-Small model for 3D occupancy mapping.

    Input: (batch, doppler, azimuth, elevation, range, 2)
        - doppler: 64
        - azimuth: 8
        - elevation: 2
        - range: 256
        - channels: 2 (I/Q)

    Output: (batch, elevation, azimuth, range)
        - elevation: 64
        - azimuth: 128
        - range: 64

    ~29M parameters for GRT-small variant.
    """

    def __init__(self):
        super().__init__()

        dim = 512
        layers = 4

        # Create encoder - stored as "tokenizer" + "encoder" in checkpoint
        # But we organize logically here and handle mapping in load_checkpoint
        self.tokenizer = GRTEncoder(
            layers=layers,
            dim=dim,
            ff_ratio=4.0,
            head_dim=64,
            dropout=0.1,
            activation="GELU",
            patch=[2, 8, 2, 4],
            pos_scale=[1.0, 1.0, 1.0, 1.0],
            global_scale=16.0,
            input_channels=2,
            positions="nd",
        )

        # Create decoder wrapper
        self.decoder = nn.Module()
        self.decoder.occ3d = GRTDecoder3D(
            key="map",
            layers=layers,
            dim=dim,
            ff_ratio=4.0,
            head_dim=64,
            dropout=0.1,
            activation="GELU",
            shape=[64, 128, 64],
            pos_scale=[1.0, 1.0, 1.0],
            global_scale=16.0,
            patch=[8, 8, 8],
            out_dim=0,
            positions="nd",
            mode="last",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encode
        encoded = self.tokenizer(x)

        # Decode
        output = self.decoder.occ3d(encoded)

        # Return just the occupancy map tensor
        return output["map"]


