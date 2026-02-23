import pytest
import torch
from ltxv_trainer.ltxv_utils import pack_latents

def test_pack_latents_shapes():
    """Test that pack_latents produces the correct output shape."""
    # B, C, F, H, W
    B, C, F, H, W = 1, 128, 8, 32, 32
    latents = torch.randn(B, C, F, H, W)

    # Test case 1: spatial_patch_size=1, temporal_patch_size=1
    packed = pack_latents(latents, spatial_patch_size=1, temporal_patch_size=1)
    # Expected shape: [B, L, D]
    # L = F * H * W = 8 * 32 * 32 = 8192
    # D = C * 1 * 1 * 1 = 128
    assert packed.shape == (B, F * H * W, C)

    # Test case 2: spatial_patch_size=2, temporal_patch_size=2
    packed = pack_latents(latents, spatial_patch_size=2, temporal_patch_size=2)
    # L = (F/2) * (H/2) * (W/2) = 4 * 16 * 16 = 1024
    # D = C * 2 * 2 * 2 = 128 * 8 = 1024
    assert packed.shape == (B, (F // 2) * (H // 2) * (W // 2), C * 2 * 2 * 2)

    # Test case 3: spatial_patch_size=4, temporal_patch_size=2
    packed = pack_latents(latents, spatial_patch_size=4, temporal_patch_size=2)
    # L = (F/2) * (H/4) * (W/4) = 4 * 8 * 8 = 256
    # D = C * 2 * 4 * 4 = 128 * 32 = 4096
    assert packed.shape == (B, (F // 2) * (H // 4) * (W // 4), C * 2 * 4 * 4)

def test_pack_latents_values():
    """Test that pack_latents permutes values correctly."""
    # Create a small tensor to track values easily
    # B=1, C=1, F=2, H=2, W=2
    # Sequential values 0 to 7
    latents = torch.arange(8).reshape(1, 1, 2, 2, 2).float()

    # spatial_patch_size=2, temporal_patch_size=2
    # This should result in a single patch containing all elements
    packed = pack_latents(latents, spatial_patch_size=2, temporal_patch_size=2)

    # Output shape should be [1, 1, 8]
    assert packed.shape == (1, 1, 8)

    # The order of packing:
    # reshape: [B, C, F/tp, tp, H/sp, sp, W/sp, sp]
    # -> [1, 1, 1, 2, 1, 2, 1, 2]
    # permute: [B, F/tp, H/sp, W/sp, C, tp, sp, sp]
    # -> [1, 1, 1, 1, 1, 2, 2, 2]
    # flatten: [1, 1, 1, 1, 8] -> [1, 1, 8]

    # The values should be ordered by C, tp, sp, sp
    # Since C=1, it iterates over tp, then sp(h), then sp(w)
    # Original layout:
    # F0:
    #   H0: W0(0), W1(1)
    #   H1: W0(2), W1(3)
    # F1:
    #   H0: W0(4), W1(5)
    #   H1: W0(6), W1(7)

    # Packed sequence expected:
    # tp=0 (F0), sp=0 (H0), sp=0 (W0) -> 0
    # tp=0 (F0), sp=0 (H0), sp=1 (W1) -> 1
    # tp=0 (F0), sp=1 (H1), sp=0 (W0) -> 2
    # tp=0 (F0), sp=1 (H1), sp=1 (W1) -> 3
    # tp=1 (F1), sp=0 (H0), sp=0 (W0) -> 4
    # ...

    expected = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]]).float()
    assert torch.equal(packed, expected)

def test_pack_latents_invalid_shapes():
    """Test that pack_latents raises RuntimeError for invalid shapes."""
    B, C, F, H, W = 1, 128, 8, 32, 32
    latents = torch.randn(B, C, F, H, W)

    # F=8 not divisible by 3
    with pytest.raises(RuntimeError):
        pack_latents(latents, spatial_patch_size=1, temporal_patch_size=3)

    # H=32 not divisible by 5
    with pytest.raises(RuntimeError):
        pack_latents(latents, spatial_patch_size=5, temporal_patch_size=1)

def test_pack_latents_identity():
    """Test identity transformation (patch size 1)."""
    B, C, F, H, W = 1, 4, 2, 2, 2
    latents = torch.randn(B, C, F, H, W)

    packed = pack_latents(latents, spatial_patch_size=1, temporal_patch_size=1)

    # Should be equivalent to permute(0, 2, 3, 4, 1).flatten(1, 3) ?
    # Let's trace logic:
    # reshape -> [B, C, F, 1, H, 1, W, 1]
    # permute -> [B, F, H, W, C, 1, 1, 1]
    # flatten -> [B, F, H, W, C]
    # flatten(1, 3) -> [B, F*H*W, C]

    expected = latents.permute(0, 2, 3, 4, 1).reshape(B, F*H*W, C)
    assert torch.allclose(packed, expected)
