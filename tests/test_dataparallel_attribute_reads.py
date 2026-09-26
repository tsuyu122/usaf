"""nn.DataParallel breaks attribute reads, not just module-name lookups.

There are two separate ways the wrapper bites, and they fail at different
moments far apart in a run.

  - named_modules() prefixes every name with 'module.', so matching a set of
    exact module names finds nothing and no gradient hook is installed. The run
    trains nothing and reports Complete.

  - an ordinary attribute like model.config is simply not there. The wrapper
    forwards module calls (__call__, named_parameters) because it subclasses
    nn.Module and nn.Module.__getattr__ searches the wrapper's own parameters,
    buffers and submodules - none of which is the inner model's config. So the
    read raises AttributeError the moment it happens.

ZAYA1-8B on a Kaggle T4 hit both in one run: it loaded, sized 14 trainable
layers, and then died asking the wrapper for its config.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = ROOT / 'usaf' / 'train.py'

pytestmark = pytest.mark.skipif(not TRAIN.is_file(), reason='trainer absent')


def _tree():
    return ast.parse(TRAIN.read_text(encoding='utf-8'))


def test_the_trainer_never_reads_an_attribute_off_the_wrapped_model():
    """Every model.<attr> left in the trainer is a DataParallel trap."""
    allowed = {
        # Forwarded by nn.Module.__getattr__ on the wrapper.
        'module', 'named_modules', 'named_parameters', 'named_buffers',
        'parameters', 'buffers', 'modules', 'train', 'eval', 'state_dict',
        'load_state_dict', 'to', 'type', 'forward',
    }
    bad = set()
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != 'model':
            continue
        if node.attr not in allowed:
            bad.add(node.attr)
    assert not bad, (
    'these reads go through a DataParallel wrapper and will raise there: '
    + ', '.join(sorted(bad))
    )


def test_the_wrapper_really_does_hide_the_config():
    """Otherwise the test above is asserting something that is not true."""
    torch = pytest.importorskip('torch')
    nn = torch.nn

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {'hidden_size': 8}
            self.lin = nn.Linear(8, 8)

    inner = Inner()
    assert inner.config['hidden_size'] == 8
    wrapped = nn.DataParallel(inner)
    with pytest.raises(AttributeError):
        _ = wrapped.config
    # The calls that do get forwarded, to show the split is real: the same
    # parameter is reachable through the wrapper, under a different name.
    inner_names = {n for n, _ in inner.named_parameters()}
    wrapped_names = {n for n, _ in wrapped.named_parameters()}
    assert inner_names, inner_names
    assert wrapped_names and wrapped_names != inner_names, (
        'if the wrapper stopped renaming, the module-name half of',
        'this suite is no longer testing anything',
    )

