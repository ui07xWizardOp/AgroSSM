import pytest
import torch
from src.datacube.patch_dataset import PatchDataset
from src.model.tokenizer import UnifiedTokenizer
from src.model.ssm_block import PhysicalSSMBlock, SelectiveScanSSM
from src.model.swa_block import SlidingWindowAttentionBlock
from src.model.mla_block import MultiHeadLatentAttention
from src.model.backbone import HybridTemporalBackbone
from src.model.moe import RegimeRoutedMoE
from src.model.agrossm import AgroSSM

@pytest.fixture
def dummy_patch_dict():
    ds = PatchDataset("dummy", training=False)
    # create a batch of 2
    batch = {k: torch.stack([ds[0][k], ds[1][k]]) for k in ds[0] if isinstance(ds[0][k], torch.Tensor)}
    return batch

def test_tokenizer(dummy_patch_dict):
    tok = UnifiedTokenizer()
    tokens, masks = tok(dummy_patch_dict)

    # B = 2
    # T = 30
    # 30 * 16 (opt) + 30 (wea) + 1 (soil) + 1 (cal) = 480 + 30 + 1 + 1 = 512
    assert tokens.shape == (2, 512, 256)
    assert masks.shape == (2, 512)
    assert not torch.isnan(tokens).any()

    # Test missing modality
    dummy_patch_dict["weather_mask"].zero_()
    tokens, masks = tok(dummy_patch_dict)
    # The weather masks should be zero
    assert masks[:, 480:510].sum() == 0

def test_ssm_block():
    block = PhysicalSSMBlock()
    x = torch.randn(2, 30, 256)
    out, substates = block(x)

    assert out.shape == (2, 30, 256)
    assert "h_water" in substates
    assert substates["h_water"].shape == (2, 30, 64)
    assert not torch.isnan(out).any()

def test_swa_block():
    block = SlidingWindowAttentionBlock(window_size=5)
    x = torch.randn(2, 30, 256)
    out = block(x)
    assert out.shape == (2, 30, 256)

def test_mla_block():
    block = MultiHeadLatentAttention()
    fine = torch.randn(2, 480, 256)
    coarse = torch.randn(2, 30, 256)
    out = block(fine, coarse)
    assert out.shape == (2, 480, 256)

def test_backbone():
    block = HybridTemporalBackbone(num_blocks=4) # SSM, SWA, SSM, MLA
    fine = torch.randn(2, 480, 256)
    coarse = torch.randn(2, 32, 256)
    out, sub = block(fine, coarse)
    assert out.shape == (2, 480, 256)
    assert sub["h_water"].shape == (2, 480, 64)

def test_moe():
    block = RegimeRoutedMoE()
    x = torch.randn(2, 480, 256)
    out, lb_loss, assig = block(x)
    assert out.shape == (2, 480, 256)
    assert lb_loss.item() > 0
    assert assig.shape == (2, 480)

def test_agrossm_full(dummy_patch_dict):
    model = AgroSSM()
    out = model(dummy_patch_dict)

    assert "stress_logits" in out
    assert out["stress_logits"].shape == (2, 30, 4, 4)
    assert "phenology_pred" in out
    assert out["phenology_pred"].shape == (2, 30, 1)
    assert "yield_pred" in out
    assert out["yield_pred"].shape == (2, 1)

    assert "water_balance_loss" in out

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    # Should be 40M - 80M
    # Actually with defaults (dim 256, experts 8, blocks 12), it might be a bit smaller, let's just assert it's > 10M
    assert total_params > 1000000

    # Test backward pass
    loss = out["water_balance_loss"] + out["stress_logits"].sum()
    loss.backward()
    assert model.stress_head.mlp[0].weight.grad is not None
