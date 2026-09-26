import ast
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1] / "usaf" / "train.py"


def _function(name):
    tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {TRAIN}")


def _guards(node):
    """Names of context managers wrapping any statement under *node*."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.With):
            for item in sub.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    func = call.func
                    parts = []
                    while isinstance(func, ast.Attribute):
                        parts.append(func.attr)
                        func = func.value
                    if isinstance(func, ast.Name):
                        parts.append(func.id)
                    out.add(".".join(reversed(parts)))
    return out


def test_the_importance_forward_runs_without_building_a_graph():
    # The importance pass replays every layer from its stored input in reverse
    # and threads the gradient through by hand, so the forward graph it used to
    # build was never differentiated. It still existed, and it pinned one
    # layer's worth of expert weights per layer simultaneously, because
    # autograd holds a matmul weight until that matmul is differentiated. That
    # is invisible on a model small enough to fit and is the whole card on one
    # that is not, and it made the out-of-memory retry useless: the peak was
    # every layer at once regardless of how many were trainable.
    fn = _function("fwd_bwd_imp")
    assert "torch.no_grad" in _guards(fn), _guards(fn)


def test_the_training_forward_also_runs_without_building_a_graph():
    fn = _function("fwd_bwd")
    assert "torch.no_grad" in _guards(fn), _guards(fn)

