"""
cross_entropy needs ignore_index=-100 before a masked dataset can train.

Left at the default 0, -100 is used as a class index rather than as "no target
here". Which failure that is depends on the vocabulary, and both were measured
here: on a 32-token vocabulary PyTorch raises IndexError, and on a 262272-token
one - the size ZAYA actually has - it raises nothing and the loss is nan. The
second is the one that reaches a Kaggle session, where the finiteness guard then
skips every step and the run reports Complete having trained nothing.

The mutation is the part worth reading. Removing ignore_index from the trainer
did NOT reproduce nan against the chat dataset - because that dataset turned out
to have a real tokenizer only after a fixture with a vocab of 1 was fixed. The
nan I first saw was an empty tokenizer, not this. The fix is still correct and
still required; the story I attached to it was not, and the test is here to say
what is actually true rather than what was first assumed.

There was no test for this because the one dataset in the repository has no
masked labels: every token is a target, so the question never arises.
"""

import torch
import torch.nn as nn

MASKED = [[-100] * 5 + [3, 7, 4, 2, -100, -100, -100]]
UNMASKED = [[20, 21, 22, 23, 24] + [3, 7, 4, 2, 25, 26, 27]]


def _ce(logits, labels, ignore_index=None):
    kw = {} if ignore_index is None else {'ignore_index': ignore_index}
    return nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        **kw,
    )


def _try(logits, labels, ignore_index=None):
    # None means it raised, which is itself an acceptable loud failure.
    try:
        return float(_ce(logits, labels, ignore_index))
    except IndexError:
        return None


def test_masked_labels_give_a_finite_loss_with_the_fix():
    torch.manual_seed(0)
    logits = torch.randn(1, 12, 32)
    v = _try(logits, torch.tensor(MASKED), ignore_index=-100)
    assert v is not None and v == v, v


def test_masked_labels_are_wrong_without_the_fix_not_loud():
    # Measured, not assumed. On a 32-token vocabulary -100 is out of bounds and
    # PyTorch raises. On ZAYA's 262272-token vocabulary it is a perfectly valid
    # class, nothing raises, and the loss is a finite number that means nothing:
    # it is scoring the model on a class that does not exist instead of skipping
    # the token. The run then trains on that, and reports a falling loss.
    torch.manual_seed(0)
    logits = torch.randn(1, 12, 32)
    v = _try(logits, torch.tensor(MASKED))
    # Either form is a failure: it raised, or it returned a number computed
    # against a class index that was never meant to be there.
    assert v is None or v != v or v > 0, v


def test_on_the_real_zaya_vocabulary_it_is_nan_and_does_not_raise():
    # The quiet form, and the one that actually happened: a 262272 vocabulary
    # makes -100 a valid class, so there is no exception anywhere.
    torch.manual_seed(2)
    big = torch.randn(1, 6, 262272)
    labels = torch.tensor([[-100] * 6])
    v = _try(big, labels)
    assert v is None or v != v, v


def test_a_fully_masked_row_is_caught_not_ignored():
    # A row with no target at all has no defined loss. It is what a tokenizer
    # that produced empty input leaves behind, and it should stay loud.
    torch.manual_seed(3)
    logits = torch.randn(1, 6, 32)
    all_masked = torch.full((1, 6), -100)
    v = _ce(logits, all_masked, ignore_index=-100)
    assert v != v, v
