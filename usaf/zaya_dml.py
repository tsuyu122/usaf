"""DML-safe forward replacement for the ZAYA MoE experts.

ZAYA was detected, downloaded, quantised and trained - and trained nothing.
That is the whole bug, and none of the existing checks could see it:

  * ``detect_model`` reads the ZAYA config, so the architecture prints right,
    the expert prefix is right, and the quantiser writes the right tensors.
  * ``_expert_modules_by_name`` finds ``model.layers.{i}.mlp.experts``, so the
    sparse-gradient capture is installed on real modules and the run does not
    stop.
  * The capture code lives *inside* the patched forward. The patch replaced
    ``Qwen3MoeExperts.forward`` and ``Qwen3MoeSparseMoeBlock.forward``, and
    ZAYA has its own ``ZayaExperts`` and ``ZayaSparseMoeBlock``, both plain
    ``nn.Module`` subclasses with no inheritance from the Qwen3 ones. Nothing
    was patched, so no hook ever fired, ``TopKImportanceStore.select`` returned
    an empty dict, and the run printed:

        Active: 0/0 (0.0000%)
        Optimizer: 0.0MB

    and went on to train the 1163 non-expert parameters, with the loss
    falling the whole time. A run that looks like training and is not.

Only the experts container is replaced. The MoE *block* is left alone on
purpose: ZAYA returns a tuple and threads router state from one decoder layer
to the next, and the Qwen3 block forward returns a single tensor and takes no
router state. Substituting it would break the model in a way that is louder
than the bug being fixed.

The replacement is ``dml_qwen3_experts_forward`` unchanged, because the two
containers are laid out identically - ``num_experts``, ``gate_up_proj``
``[E, 2*inter, hidden]``, ``down_proj`` ``[E, hidden, inter]``, ``act_fn`` -
and ZAYA calls ``self.experts(hs, router_indices, router_probs)`` with three
positional arguments, which is exactly the shape that forward already takes.
"""
from usaf.qwen3moe_dml import dml_qwen3_experts_forward


def patch_zaya_for_dml():
    """Replace ZayaExperts.forward with the dense-masked sparse-capturing one.

    A no-op when the Zyphra fork is not installed, because ZayaForCausalLM
    does not exist in stock transformers. Importing usaf must never require a
    fork.
    """
    try:
        from transformers.models.zaya import modeling_zaya
    except ImportError:
        return None

    cls = modeling_zaya.ZayaExperts
    if not hasattr(cls, "_usaf_original"):
        cls._usaf_original = cls.forward
    cls.forward = dml_qwen3_experts_forward
    return modeling_zaya


def unpatch_zaya_for_dml():
    """Restore the stock ZayaExperts.forward."""
    try:
        from transformers.models.zaya import modeling_zaya
    except ImportError:
        return

    cls = modeling_zaya.ZayaExperts
    if hasattr(cls, "_usaf_original"):
        cls.forward = cls._usaf_original
        del cls._usaf_original
