from pathlib import Path

import torch
from diffusers import AutoencoderTiny


class VAE:
    # Bundled with the artifact so inference does not depend on a user cache
    # or network access.
    MODEL_PATH = Path(__file__).resolve().parents[4] / "checkpoints" / "third_party" / "taesd"

    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        self.scaling_factor = 0.18215
        self._init_vae()

    def _init_vae(self):
        # When loading a model with known architecture mismatches,
        # it's safer to disable low_cpu_mem_usage. This prevents
        # the creation of "meta" tensors that cause errors on .to(device).
        vae = AutoencoderTiny.from_pretrained(
            str(self.MODEL_PATH),
            local_files_only=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device, memory_format=torch.channels_last)
        print("Loaded bundled TAESD.")

        for param in vae.parameters():
            param.requires_grad = False

        # Compile the entire VAE module after it's on the correct device
        self.vae = vae

    def encode_latent(self, input_neg_one_to_one):
        """Encodes an input tensor from the range [-1, 1]."""
        input_tensor = input_neg_one_to_one.to(
            self.device, dtype=self.torch_dtype, memory_format=torch.channels_last
        )
        # with torch.no_grad():
        #     # Call the compiled VAE's encode method
        latents = self.vae.encode(input_tensor).latents * self.scaling_factor

        return latents

    def decode_latent(self, latents):
        """
        Decodes latents back into image space.

        Note: VAE parameters are frozen (requires_grad=False), but gradients
        can still flow through this operation for backpropagation to upstream models.
        """
        latents = latents.to(
            self.device, dtype=self.torch_dtype, memory_format=torch.channels_last
        )
        # Decode without blocking gradients (VAE is frozen via requires_grad=False)
        decoded = self.vae.decode(latents / self.scaling_factor).sample

        # Output in [0, 1] range
        decoded_zero_one = (decoded * 0.5 + 0.5).clamp(0, 1)
        # Output in [-1, 1] range
        decoded_neg_pos = decoded.clamp(-1, 1)

        return decoded_zero_one, decoded_neg_pos


