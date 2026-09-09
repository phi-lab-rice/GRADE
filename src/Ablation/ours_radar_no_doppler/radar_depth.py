import torch
import torch.nn as nn
from typing import Tuple


class RadarPatchEmbed(nn.Module):
    """
    Radar Spectrum Patch Embedding Layer.

    Takes 5D radar spectrum data and converts it into patch embeddings:
    1. Input: [B, 2, 256, 64, 8, 2] where channels are (magnitude, phase)
    2. Patchifies along range and doppler dimensions
    3. Outputs: [B, num_patches, embed_dim] where num_patches = 2048

    Patch extraction:
    - Range dimension (256): patch_size=4, stride=4 -> 64 patches
    - Doppler dimension (64): patch_size=2, stride=2 -> 32 patches
    - Total patches: 64 × 32 = 2048
    - Each patch: [4 range × 2 doppler × 8 elevation × 2 azimuth] × 2 channels = 256 features
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (
            256,
            64,
            8,
            2,
        ),  # (Range, Doppler, Elevation, Azimuth)
        patch_size: Tuple[int, int, int, int] = (
            4,
            2,
            8,
            2,
        ),  # (Range, Doppler, Elevation, Azimuth)
        stride: Tuple[int, int] = (4, 2),  # (Range, Doppler)
        embed_dim: int = 256,
        in_channels: int = 2,  # magnitude + phase
    ):
        super().__init__()

        self.input_shape = input_shape
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # Calculate number of patches
        range_dim, doppler_dim, elev_dim, azim_dim = input_shape
        patch_range, patch_doppler, patch_elev, patch_azim = patch_size
        stride_range, stride_doppler = stride

        self.num_patches_range = (range_dim - patch_range) // stride_range + 1  # 64
        self.num_patches_doppler = (
            doppler_dim - patch_doppler
        ) // stride_doppler + 1  # 32
        self.num_patches = self.num_patches_range * self.num_patches_doppler  # 2048

        # Each patch has: patch_range × patch_doppler × patch_elev × patch_azim features per channel
        patch_volume = (
            patch_range * patch_doppler * patch_elev * patch_azim
        )  # 4×2×8×2 = 128
        self.patch_features = patch_volume * in_channels  # 128 × 2 = 256

        # Linear projection from patch features to embedding dimension
        self.proj = nn.Linear(self.patch_features, embed_dim)

        print(f"Radar Patch Embedding Configuration:")
        print(
            f"  Input shape: [B, {in_channels}, {range_dim}, {doppler_dim}, {elev_dim}, {azim_dim}]"
        )
        print(f"  Patch size: {patch_size}")
        print(f"  Stride: {stride}")
        print(
            f"  Number of patches (range × doppler): {self.num_patches_range} × {self.num_patches_doppler} = {self.num_patches}"
        )
        print(f"  Patch features per channel: {patch_volume}")
        print(f"  Total patch features (mag+phase): {self.patch_features}")
        print(f"  Embedding dimension: {embed_dim}")

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from radar spectrum data.

        Args:
            x: [B, 2, 256, 64, 8, 2] (magnitude + phase channels)

        Returns:
            patches: [B, num_patches, patch_features]
        """
        batch_size = x.shape[0]
        x_mag = x[:, 0]  # [B, 256, 64, 8, 2]
        x_phase = x[:, 1]  # [B, 256, 64, 8, 2]

        all_patches = []

        # Extract patches with stride along range and doppler dimensions
        for i in range(self.num_patches_range):
            for j in range(self.num_patches_doppler):
                start_range = i * self.stride[0]
                end_range = start_range + self.patch_size[0]
                start_doppler = j * self.stride[1]
                end_doppler = start_doppler + self.patch_size[1]

                # Extract patch from both channels
                patch_mag = x_mag[
                    :, start_range:end_range, start_doppler:end_doppler, :, :
                ]
                patch_phase = x_phase[
                    :, start_range:end_range, start_doppler:end_doppler, :, :
                ]

                # Flatten patches
                patch_mag_flat = patch_mag.flatten(1)  # [B, 128]
                patch_phase_flat = patch_phase.flatten(1)  # [B, 128]

                # Interleave magnitude and phase features
                patch_interleaved = torch.stack(
                    [patch_mag_flat, patch_phase_flat], dim=-1
                )
                patch_interleaved = patch_interleaved.flatten(1, -1)  # [B, 256]

                all_patches.append(patch_interleaved)

        # Stack all patches: [B, num_patches, patch_features]
        all_patches = torch.stack(all_patches, dim=1)
        return all_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, 2, 256, 64, 8, 2]

        Returns:
            embeddings: [B, num_patches, embed_dim]
        """
        # Extract patches: [B, 2048, 256]
        patches = self.extract_patches(x)

        # Project to embedding dimension: [B, 2048, embed_dim]
        embeddings = self.proj(patches)

        return embeddings


class RadarEncoder(nn.Module):
    """
    Radar Vision Transformer (ViT) Encoder.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (256, 64, 8, 2),
        patch_size: Tuple[int, int, int, int] = (4, 2, 8, 2),
        stride: Tuple[int, int] = (4, 2),
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Patch embedding layer
        self.patch_embed = RadarPatchEmbed(
            input_shape=input_shape,
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            in_channels=2,
        )

        self.num_patches = self.patch_embed.num_patches

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize patch embedding projection
        if hasattr(self.patch_embed.proj, "weight"):
            nn.init.xavier_uniform_(self.patch_embed.proj.weight)
            if self.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.patch_embed.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with self-attention and feedforward"""

    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class DepthDecoder(nn.Module):
    """
    Hybrid Transformer+CNN decoder for depth image generation.

    Input: [batch_size, num_patches=2048, embed_dim=512]
    Output: [batch_size, 1, height=128, width=256]

    Architecture:
    1. Transformer decoder blocks (4 layers)
    2. Reshape to 2D feature map (64x32)
    3. CNN upsampling stages (64x32 -> 128x256)
    """

    def __init__(
        self,
        embed_dim=256,
        num_patches=2048,
        patch_grid_size=(64, 32),  # Spatial structure from radar encoder
        num_decoder_blocks=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        output_height=128,
        output_width=256,
        output_channels=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_grid_size = patch_grid_size  # (64, 32) spatial grid
        self.output_height = output_height
        self.output_width = output_width
        self.output_channels = output_channels

        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_decoder_blocks)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Projection to intermediate feature map
        # From 64x32x256 to 64x32x128 (reduce dimension for upsampling)
        self.feature_proj = nn.Conv2d(embed_dim, 128, kernel_size=1)

        # Upsampling network: 64x32 -> 128x256
        # Start from 64x32 (range x doppler), upsample to 128x256
        self.upsample = nn.Sequential(
            # Upsample doppler dimension: 64x32 -> 64x64
            nn.Upsample(scale_factor=(1, 2), mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Upsample both dimensions: 64x64 -> 128x128
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Upsample width dimension: 128x128 -> 128x256
            nn.Upsample(scale_factor=(1, 2), mode="bilinear", align_corners=False),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_patches, embed_dim]

        Returns:
            depth: [batch_size, output_channels, output_height, output_width]
        """
        batch_size = x.shape[0]

        # Apply transformer decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape to spatial dimensions: [B, 2048, 512] -> [B, 64, 32, 512]
        x = x.reshape(
            batch_size,
            self.patch_grid_size[0],  # 64 (range)
            self.patch_grid_size[1],  # 32 (doppler)
            self.embed_dim,
        )

        # Permute to channel-first: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        # Shape: [batch, 512, 64, 32]

        # Project features
        x = self.feature_proj(x)
        # Shape: [batch, 128, 64, 32]

        # Upsample to target resolution
        depth = self.upsample(x)
        # Shape: [batch, 1, 128, 256]

        return depth


class RadarDepth(nn.Module):
    """
    End-to-end Radar to Depth model (Doppler-as-Channels).
    """

    def __init__(
        self,
        # Encoder args
        input_shape: Tuple[int, int, int, int] = (256, 64, 8, 2),
        patch_size: Tuple[int, int, int, int] = (4, 2, 8, 2),
        stride: Tuple[int, int] = (4, 2),
        embed_dim: int = 256,
        encoder_num_heads: int = 8,
        encoder_num_layers: int = 4,
        encoder_mlp_ratio: float = 4.0,
        encoder_dropout: float = 0.1,
        # Decoder args
        decoder_num_blocks: int = 4,
        decoder_num_heads: int = 8,
        output_height: int = 128,
        output_width: int = 256,
    ):
        super().__init__()

        self.encoder = RadarEncoder(
            input_shape=input_shape,
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            num_heads=encoder_num_heads,
            num_layers=encoder_num_layers,
            mlp_ratio=encoder_mlp_ratio,
            dropout=encoder_dropout,
        )

        # Get patch info from encoder
        num_patches = self.encoder.num_patches  # 64

        self.decoder = DepthDecoder(
            embed_dim=embed_dim,
            num_patches=num_patches,
            num_decoder_blocks=decoder_num_blocks,
            num_heads=decoder_num_heads,
            output_height=output_height,
            output_width=output_width,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def create_radar_encoder(*args, **kwargs):
    return RadarEncoder(*args, **kwargs)


