import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _no_dml_patch_survives_a_test():
    """Every test gets the stock expert forwards back, whatever it did.

    setup_device installs the DML forward replacements process-wide and they are
    not scoped to a run: a test that calls the trainer in-process leaves them
    installed, and the next test then quietly exercises the dense-masked path
    instead of the stock one. That showed up as a dtype error two files away
    from the cause, and as a signature test comparing the patch with itself.

    Doing it per test file means remembering. Doing it here means not having to.
    """
    yield
    from usaf.mixtral_dml import unpatch_mixtral_for_dml
    from usaf.olmoe_dml import unpatch_olmoe_for_dml
    from usaf.qwen3moe_dml import unpatch_qwen3moe_for_dml

    unpatch_qwen3moe_for_dml()
    unpatch_olmoe_for_dml()
    unpatch_mixtral_for_dml()
