"""The checkpoint layout that save_pretrained writes, and that every real model ships.

This is the bug the first Kaggle run found. usaf.quantize looked for the stacked
expert tensors - gate_up_proj of shape (E, 2*inter, hidden) - by name. That is
what the model holds in memory once from_pretrained has run, and it is not
what transformers writes to disk. A checkpoint written by save_pretrained has
one tensor per expert:

    model.layers.0.mlp.experts.0.gate_proj.weight
    model.layers.0.mlp.experts.0.up_proj.weight
    model.layers.0.mlp.experts.0.down_proj.weight

so the quantizer found none of the eight tensors it asked for and exited. The
fixture in tests/ happened to carry the stacked layout already, so the local
test passed - the one case in a long line of them where the fixture hid the
defect rather than the code being right.

The two assertions below are the ones that matter. The first says the
reconstruction is exact, so a zero difference is evidence and not a
coincidence of symmetric projections. The second says both routes produce
bit-identical 4-bit output, which holds whatever the quantizer error is and
so it cannot be confused with it.
"""
import os
import subprocess
import sys

import pytest

E2E = "C:/Users/hm/Projects/e2e"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(E2E, "tiny-moe")),
    reason="e2e fixtures absent",
)


@pytest.fixture(scope="module")
def split_model(tmp_path_factory):
    """A Qwen3-MoE saved by save_pretrained, i.e. the per-expert layout."""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    work = tmp_path_factory.mktemp("split")
    d = work / "split"
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(
        vocab_size=512,
        hidden_size=512,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=64,
        num_experts=8,
        num_experts_per_tok=2,
        num_local_experts=8,
        decoder_sparse_step=1,
        norm_topk_prob=False,
        tie_word_embeddings=False,
    )
    Qwen3MoeForCausalLM(cfg).save_pretrained(d)

    # the layout is the whole point of the fixture, so assert it
    from safetensors import safe_open

    with safe_open(d / "model.safetensors", framework="pt") as f:
        keys = list(f.keys())
    assert any(".experts.0.gate_proj.weight" in k for k in keys), keys[:5]
    assert not any(k.endswith("experts.gate_up_proj") for k in keys), (
        "save_pretrained produced the stacked layout; the fixture would not "
        "exercise the bug it exists to cover"
    )
    return d


def test_the_stacked_names_are_absent_from_the_checkpoint(split_model):
    '''Document why the module needs a reconstruction at all.'''
    from safetensors import safe_open

    with safe_open(split_model / "model.safetensors", framework="pt") as f:
        keys = list(f.keys())
    experts = sorted(k for k in keys if ".experts" in k)
    assert experts, "the fixture has no expert tensors"

    stacked = [k for k in experts if k.endswith(("gate_up_proj", "down_proj"))]
    assert not stacked, f'stacked present, covers nothing: {stacked}'

    per_expert = [k for k in experts if ".gate_proj." in k
                  or ".up_proj." in k or ".down_proj." in k]
    assert len(per_expert) == len(experts), experts[:3]



def test_the_reconstruction_is_exact(split_model):
    """gate_up_proj is exactly stack(cat(gate, up)). Zero, not close to zero."""
    import torch
    from safetensors import safe_open
    from transformers import Qwen3MoeForCausalLM

    m = Qwen3MoeForCausalLM.from_pretrained(split_model)
    params = dict(m.named_parameters())
    n_experts = 8

    with safe_open(split_model / "model.safetensors", framework="pt") as f:
        for li in range(2):
            pre = f"model.layers.{li}.mlp.experts"
            gates = [f.get_tensor(f"{pre}.{e}.gate_proj.weight")
                     for e in range(n_experts)]
            ups = [f.get_tensor(f"{pre}.{e}.up_proj.weight")
                   for e in range(n_experts)]
            downs = [f.get_tensor(f"{pre}.{e}.down_proj.weight")
                     for e in range(n_experts)]
            gu = torch.stack([torch.cat([gates[e], ups[e]], dim=0)
                             for e in range(n_experts)])
            dp = torch.stack(downs)

            assert float((gu - params[f"{pre}.gate_up_proj"].data)
                         .abs().max()) == 0.0, "gate/up order or stacking is wrong"
            assert float((dp - params[f"{pre}.down_proj"].data)
                         .abs().max()) == 0.0, "down_proj stacking is wrong"

            # the reverse order has to be different, otherwise a zero above
            # proves nothing
            up_first = torch.stack([torch.cat([ups[e], gates[e]], dim=0)
                                    for e in range(n_experts)])
            assert float((gu - up_first).abs().max()) > 0, (
                "gate and up are identical, so the ordering is untested"
            )


def test_both_routes_produce_bit_identical_quantization(split_model):
    """Isolates the reconstruction from the quantizer error entirely."""
    import torch
    from safetensors import safe_open
    from transformers import Qwen3MoeForCausalLM

    from usaf.quantization import quantize_state_dict

    m = Qwen3MoeForCausalLM.from_pretrained(split_model)
    params = dict(m.named_parameters())
    n_experts = 8

    with safe_open(split_model / "model.safetensors", framework="pt") as f:
        pre = "model.layers.0.mlp.experts"
        gates = [f.get_tensor(f"{pre}.{e}.gate_proj.weight")
                 for e in range(n_experts)]
        ups = [f.get_tensor(f"{pre}.{e}.up_proj.weight")
               for e in range(n_experts)]
        rebuilt = {
            f"{pre}.gate_up_proj": torch.stack(
                [torch.cat([gates[e], ups[e]], dim=0) for e in range(n_experts)]
            ),
        }

    a = quantize_state_dict(rebuilt, group_size=128)
    b = quantize_state_dict({k: params[k].data for k in rebuilt}, group_size=128)

    for name in rebuilt:
        assert float((a[name]["q"].float() - b[name]["q"].float())
                     .abs().max()) == 0.0, name
        assert float((a[name]["s"].float() - b[name]["s"].float())
                     .abs().max()) == 0.0, name


def test_the_cli_actually_quantizes_a_split_layout_model(split_model,
                                                         tmp_path):
    """End to end on the layout that broke the first Kaggle run."""
    work = tmp_path / "out"
    out = subprocess.run(
        [sys.executable, "-m", "usaf.quantize",
         "--model", str(split_model), "--out", str(work)],
        cwd=E2E,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = (out.stdout or "") + (out.stderr or "")
    assert "Traceback" not in log, log[-800:]
    assert "missing" not in log, log[-600:]
    produced = work / "experts_q4.pt"
    assert produced.exists(), log[-600:]

    import torch

    q = torch.load(produced, map_location="cpu", weights_only=False)
    assert len(q) == 4, sorted(q)
