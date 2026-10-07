import math
import torch
from torch import nn
from typing import Optional

from config import PretrainConfig

def get_patches_center_coordinates(
    num_patches_h: int, num_patches_w: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Computes the 2D coordinates of the centers of image patches, normalized to the range [-1, +1].
    The center of each patch is exactly halfway between its top-left and bottom-right corners.

    Args:
        num_patches_h (int): Number of patches along the vertical (height) axis.
        num_patches_w (int): Number of patches along the horizontal (width) axis.
        dtype (torch.dtype): The desired data type of the returned tensor.
        device (torch.device): Device on which the tensor is allocated.

    Returns:
        torch.Tensor: A tensor of shape (num_patches_h * num_patches_w, 2), where each row contains the (y, x)
            coordinates of a patch center, normalized to [-1, +1].
    """
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    # (height, width, 2) -> (height * width, 2)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    # Shift range [0, 1] to [-1, +1]
    coords = 2.0 * coords - 1.0
    return coords

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RoPE(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, "Hidden size must be divisible by num_attention_heads"
        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)  # (D/4,)
        patch_coords = get_patches_center_coordinates(
            config.image_size // config.patch_size,
            config.image_size // config.patch_size,
            dtype=torch.float32,
            device="cpu",
        )  # [N,2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("patch_coords", patch_coords, persistent=False)
        self.grid = (config.image_size // config.patch_size, config.image_size // config.patch_size)
        self.rescale = config.pos_embed_rescale

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor | None = None):
        B, _, height, width = pixel_values.shape
        grid = (height // self.config.patch_size, width // self.config.patch_size)
        N = grid[0] * grid[1]

        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            if grid == self.grid:
                patch_coords = self.patch_coords
            else: # e.g. segmentation crops: the same [-1, 1] coordinates on another grid (as NEPA)
                patch_coords = get_patches_center_coordinates(*grid, dtype=torch.float32, device=device)
            if self.training and self.rescale is not None:
                # scale coords by a log-uniform factor in [1/rescale, rescale], shared by the batch (NEPA / DINOv3)
                log_rescale = math.log(self.rescale)
                patch_coords = patch_coords * torch.empty(1, device=device).uniform_(-log_rescale, log_rescale).exp()
            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]  # [N,2,D/4]
            angles = angles.flatten(1, 2).tile(2)  # [N,D]
            cos_all = torch.cos(angles)  # [N,D]
            sin_all = torch.sin(angles)  # [N,D]

            if position_ids is None:
                position_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B,N]

            cos = cos_all[position_ids]  # [B,N,D]
            sin = sin_all[position_ids]  # [B,N,D]

        dt = pixel_values.dtype
        return cos.to(dt), sin.to(dt)

def prepend_prefix_tokens(position_embeddings: tuple[torch.Tensor, torch.Tensor], num_prefix_tokens: int = 1):
    """ Identity rotation (cos 1, sin 0) for prefix tokens such as CLS, which RoPE leaves unrotated (as NEPA) """
    cos, sin = position_embeddings # [B,N,D]
    B, _, D = cos.shape
    cos = torch.cat([cos.new_ones(B, num_prefix_tokens, D), cos], dim=1)
    sin = torch.cat([sin.new_zeros(B, num_prefix_tokens, D), sin], dim=1)
    return cos, sin

def apply_rotary_pos_emb(q: torch.Tensor, k: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor):
    # q,k: [B,H,T,D]; cos,sin: [B,N,D]
    assert cos.dim() == sin.dim() == 3, "cos and sin must have shape [B,N,D]"

    # cos / sin keep their dtype (float32), so the rotation runs in float32 even for bfloat16 q / k (as NEPA)
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]

    q = (q * cos) + (rotate_half(q) * sin)
    if k is not None:
        k = (k * cos) + (rotate_half(k) * sin)
    return q, k
