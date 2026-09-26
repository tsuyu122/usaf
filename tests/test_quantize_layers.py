"""--layers has to decide what gets quantized, not just what gets reported.

collect_expert_tensors recomputed the layer patterns from the model config
every time it ran, so main() narrowed the list, printed it, and then quantized
every layer anyway: "quantizing 2 expert tensors" followed by a file with eight
in it. On a model large enough that --layers is the difference between fitting
in memory and not, that is a flag that reports success and does nothing.

The trainer is what consumes this file, so the check is the file: which layers
it names, not what the tool said it was doing.
"""
import contextlib
import io
import os
import re

import pytest
import torch

E2E = r"C:\Users\hm\Projects\e2e"
MODEL = os.path.join(E2E, "tiny-moe")

pytestmark = pytest.mark.skipif(not os.path.isdir(MODEL),
                                reason="e2e fixture absent")


def _run(tmp_path, extra):
    from usaf.quantize import main
    out = tmp_path / "q4"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = main(["--model", MODEL, "--out", str(out)] + extra)
    assert rc == 0, buf.getvalue()[-2000:]
    d = torch.load(out / "experts_q4.pt", map_location="cpu",
                   weights_only=False)
    layers = sorted({int(re.search(r"layers\.(\d+)", k).group(1))
                     for k in d})
    return layers, len(d), buf.getvalue()


def test_layers_zero_quantizes_only_layer_zero(tmp_path):
    layers, n, out = _run(tmp_path, ["--layers", "0"])
    assert layers == [0], f"the file holds layers {layers}"


def test_the_count_it_prints_is_the_count_it_writes(tmp_path):
    """The message is what misleads, so check it against the file."""
    layers, n, out = _run(tmp_path, ["--layers", "0"])
    m = re.search(r"quantizing (\d+) expert tensors", out)
    assert m, out[-1500:]
    assert int(m.group(1)) == n, f"said {m.group(1)} tensors, wrote {n}"


def test_two_layers_are_two_layers(tmp_path):
    layers, _, _ = _run(tmp_path, ["--layers", "0,2"])
    assert layers == [0, 2], layers


def test_no_flag_means_every_layer(tmp_path):
    layers, n, out = _run(tmp_path, [])
    assert len(layers) > 1, layers
    m = re.search(r"quantizing (\d+) expert tensors", out)
    assert int(m.group(1)) == n, (m.group(1), n)
