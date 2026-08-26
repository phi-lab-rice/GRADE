from pathlib import Path

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class SDUnet:
    MODEL_PATH = Path(__file__).resolve().parents[4] / "checkpoints" / "third_party" / "marigold_unet"
    def __init__(
        self,
        use_pretrained=False,
        device="cuda",
        torch_dtype=torch.float16,
    ):
        """
        Initialize SD UNet (Marigold).

        Args:
            use_pretrained: Whether to load pretrained weights
            device: Device to load model on
            torch_dtype: Data type for model weights
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self._init_unet(use_pretrained)

    def _init_unet(self, use_pretrained):
        # The released checkpoint supplies learned weights; this bundled file
        # provides only the corresponding architecture configuration.
        model_path = str(self.MODEL_PATH)

        if use_pretrained:
            # OPTIMIZATION 1: Pass torch_dtype inside from_pretrained.
            # This loads the weights directly in the target precision (fp16/bf16),
            # preventing the CPU RAM spike of loading the full fp32 model first.
            self.unet = UNet2DConditionModel.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                local_files_only=True,
            ).to(self.device)

        else:
            # Initialize random weights from config (for training from scratch)
            config = UNet2DConditionModel.load_config(
                pretrained_model_name_or_path=model_path, local_files_only=True
            )

            # from_config usually initializes in float32 on CPU, so we must cast after.
            self.unet = UNet2DConditionModel.from_config(config).to(
                self.device, dtype=self.torch_dtype
            )


