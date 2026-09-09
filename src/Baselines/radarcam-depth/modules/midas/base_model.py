import torch
from safetensors.torch import load_file


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        self.load_state_dict(load_file(path, device="cpu"), strict=True)
