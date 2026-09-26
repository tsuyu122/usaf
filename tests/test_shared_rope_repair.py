import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "usaf"


def _guess_calls():
    """Find getattr(mod, "dim", ...) in code, not in the comments about it."""
    hits = []
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, ast.Name) or fn.id != "getattr":
                continue
            args = node.args
            if len(args) >= 2 and isinstance(args[1], ast.Constant):
                if args[1].value == "dim":
                    hits.append(f"{path.name}:{node.lineno}")
    return hits


def test_no_module_guesses_the_rotary_dimension():
    # The reconstruction guessed the rotary dim with getattr on an attribute the
    # module does not have, so it always fell through to 128 and produced an
    # inv_freq of the wrong length: the forward died with "The size of tensor
    # a (8) must match tensor b (128)". It was fixed in the training loader and
    # came back verbatim in the streaming helper. No copy may survive. Comments
    # describing the bug are not copies, so this reads the tree, not the text.
    assert not _guess_calls(), _guess_calls()


def test_the_streaming_helper_calls_the_shared_repair():
    src = (SRC / "qwen3_setup.py").read_text(encoding="utf-8")
    assert "materialize_meta_tensors(" in src

