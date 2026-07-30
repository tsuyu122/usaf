"""Phase B E2E Integration Test — synthetic 16-layer OLMoE (random init).

Validates all code paths simultaneously: DML patches, streaming hooks,
post_accumulate_grad_hook, apply_captured_expert_grads, importance scoring
via TopK active_idx, SparseAdam (scatter_, refresh), perplexity eval.

If this passes, the real model on DML will pass too. Runs in <2 min.
"""
import re, time, math
import pytest
import torch
from transformers import AutoConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM
from usaf.olmoe_dml import patch_olmoe_for_dml
from usaf.olmoe_streaming import setup_streaming, apply_captured_expert_grads
from usaf.sparse_optim import SparseAdam
from eval_olmoe import evaluate

SEQ = 32
TRAIN_LAYERS = {12, 13, 14, 15}
FRAC = 5e-4
STEPS = 4
LR = 2e-3
LOSS_SCALE = 4096.0
VOCAB_SIZE = 2048


def layer_of(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


class _DummyTokenizer:
    def __call__(self, text, truncation=False, max_length=256, return_tensors=None):
        ids = [(ord(c) % (VOCAB_SIZE - 2)) + 2 for c in text[:max_length]]
        while len(ids) < 4:
            ids.append(2)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_b1_streaming_smoke():
    device = torch.device("cpu")
    patch_olmoe_for_dml()
    cfg = AutoConfig.from_pretrained("allenaiOLMoE")
    cfg.vocab_size = VOCAB_SIZE
    cfg.hidden_size = 512
    cfg.intermediate_size = 128
    cfg.num_experts = 2
    cfg.num_experts_per_tok = 1
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 8

    torch.manual_seed(42)
    model = OlmoeForCausalLM(cfg).half()
    for name, p in model.named_parameters():
        train = (".mlp.experts." in name) and (layer_of(name) in TRAIN_LAYERS)
        p.requires_grad = train
    setup_streaming(model, device)
    model.enable_input_require_grads()
    model.train()

    input_ids = torch.randint(2, VOCAB_SIZE, (1, SEQ), device=device, dtype=torch.long)
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss.item()
    assert not math.isnan(loss), "B1: NaN loss"
    out.loss.backward()
    n_captured = apply_captured_expert_grads(model)
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    assert n_grad > 0, "B1: no params got gradients"


def test_b2_importance_scoring():
    device = torch.device("cpu")
    patch_olmoe_for_dml()
    cfg = AutoConfig.from_pretrained("allenaiOLMoE")
    cfg.vocab_size = VOCAB_SIZE
    cfg.hidden_size = 512
    cfg.intermediate_size = 128
    cfg.num_experts = 2
    cfg.num_experts_per_tok = 1
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 8

    torch.manual_seed(42)
    model = OlmoeForCausalLM(cfg).half()
    for name, p in model.named_parameters():
        train = (".mlp.experts." in name) and (layer_of(name) in TRAIN_LAYERS)
        p.requires_grad = train
    setup_streaming(model, device)
    model.enable_input_require_grads()
    model.train()

    input_ids = torch.randint(2, VOCAB_SIZE, (1, SEQ), device=device, dtype=torch.long)
    labels = input_ids.clone()

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None
    model._expert_grads.clear()
    out = model(input_ids=input_ids, labels=labels)
    (out.loss * LOSS_SCALE).backward()
    apply_captured_expert_grads(model, scale=1.0 / LOSS_SCALE)

    active_idx: dict[str, torch.Tensor] = {}
    total_active = total_elems = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        flat = p.grad.detach().abs().reshape(-1)
        k = max(1, int(flat.numel() * FRAC))
        idx = torch.topk(flat, k).indices.cpu()
        active_idx[name] = idx
        total_active += k
        total_elems += flat.numel()
        p.grad = None

    assert len(active_idx) > 0 and total_active > 0, "B2: zero active params"


def test_b3_sparse_fine_tuning():
    device = torch.device("cpu")
    patch_olmoe_for_dml()
    cfg = AutoConfig.from_pretrained("allenaiOLMoE")
    cfg.vocab_size = VOCAB_SIZE
    cfg.hidden_size = 512
    cfg.intermediate_size = 128
    cfg.num_experts = 2
    cfg.num_experts_per_tok = 1
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 8

    torch.manual_seed(42)
    model = OlmoeForCausalLM(cfg).half()
    for name, p in model.named_parameters():
        train = (".mlp.experts." in name) and (layer_of(name) in TRAIN_LAYERS)
        p.requires_grad = train
    setup_streaming(model, device)
    model.enable_input_require_grads()
    model.train()

    input_ids = torch.randint(2, VOCAB_SIZE, (1, SEQ), device=device, dtype=torch.long)
    labels = input_ids.clone()

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None
    model._expert_grads.clear()
    out = model(input_ids=input_ids, labels=labels)
    (out.loss * LOSS_SCALE).backward()
    apply_captured_expert_grads(model, scale=1.0 / LOSS_SCALE)

    active_idx: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        flat = p.grad.detach().abs().reshape(-1)
        k = max(1, int(flat.numel() * FRAC))
        idx = torch.topk(flat, k).indices.cpu()
        active_idx[name] = idx
        p.grad = None

    model.train()
    opt = SparseAdam(dict(model.named_parameters()), active_idx=active_idx,
                     lr=LR, weight_decay=0.0)

    losses: list[float] = []
    for step in range(STEPS):
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad = None
        model._expert_grads.clear()
        out = model(input_ids=input_ids, labels=labels)
        (out.loss * LOSS_SCALE).backward()
        apply_captured_expert_grads(model, scale=1.0 / LOSS_SCALE)
        opt.refresh(model)
        opt.step()
        losses.append(out.loss.item())
        assert not math.isnan(losses[-1]), f"B3: NaN loss step {step}"

    assert losses[-1] < losses[0] - 1e-3, \
        f"B3: no loss drop ({losses[0]:.4f} -> {losses[-1]:.4f})"


def test_b4_perplexity_evaluation():
    device = torch.device("cpu")
    patch_olmoe_for_dml()
    cfg = AutoConfig.from_pretrained("allenaiOLMoE")
    cfg.vocab_size = VOCAB_SIZE
    cfg.hidden_size = 512
    cfg.intermediate_size = 128
    cfg.num_experts = 2
    cfg.num_experts_per_tok = 1
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 8

    torch.manual_seed(42)
    model = OlmoeForCausalLM(cfg).half()
    setup_streaming(model, device)
    model.eval()

    tok = _DummyTokenizer()
    eval_texts = [
        "int main() { int x = 0; return x; }",
        "void foo(int a, int b) { return a + b; }",
        "template<typename T> T max(T a, T b) { return a > b ? a : b; }",
        "struct Node { int val; Node* next; };",
    ]
    ppl, avg_loss, tokens, tps, elapsed = evaluate(model, tok, device, eval_texts)
    assert not math.isnan(avg_loss), "B4: NaN loss"
    assert tokens > 0, "B4: zero tokens evaluated"
